# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence guards for the packed-cache slot-mapping Triton kernel.

The kernel replaced the eager ``where/div/remainder/copy_`` chain that every KV
cache group ran per step. Its output feeds the packed attention planes directly,
so an arithmetic-only optimisation has to stay bit-identical -- including the
sentinel rows that a full-graph replay leaves in the padded tail of the token
carrier.

Two semantics are easy to break while tuning, so both are pinned here:

1. ``ratio=1`` masks rows past ``min(query_start_loc[num_actual_reqs],
   num_actual_tokens)``. The eager chain does *not* apply that mask when the
   slot mapping is uncompressed; the kernel does, and the difference is
   deliberate. ``test_padded_tail_is_masked_for_ratio1`` pins the kernel
   behaviour rather than the eager one, so a future "align with eager" change
   cannot quietly start writing those rows.
2. ``page_size`` need not be a power of two. The optimised kernel takes a
   shift/mask path for powers of two and keeps the original div/remainder path
   otherwise; both must agree with the same reference.

Run inside the service container (``torch_npu`` is required):

    docker exec -e PYTHONPATH=/vllm-workspace/vllm-ascend dsv41-rc2 \
        python -m pytest <this file> -q
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.triton.packed_cache_slot_mapping import (  # noqa: E402
    build_packed_cache_slot_mapping,
)

DEVICE = "npu"


def _reference(slots, positions, qsl, num_reqs, num_actual_tokens, page, ratio, skip):
    """The pre-kernel eager chain, verbatim from dsa_v41.py's fallback branch.

    It applies the ``valid_end`` mask only for ``ratio == 2``. That is the
    upstream contract for the eager path and intentionally *not* what the
    kernel does; see ``test_padded_tail_is_masked_for_ratio1``.
    """
    active = slots
    if ratio == 2:
        keep = (slots >= 0) & ((slots + 1) % ratio == 0)
        active = torch.where(keep, slots // ratio, -1)
    valid = active >= 0
    if ratio == 2:
        if skip:
            valid = torch.zeros_like(valid)
        else:
            vend = torch.minimum(
                qsl[num_reqs],
                torch.tensor(num_actual_tokens, device=slots.device, dtype=qsl.dtype),
            )
            tok = torch.arange(slots.numel(), device=slots.device)
            valid = valid & (tok < vend)
            if positions is not None:
                valid = valid & (positions.remainder(2) == 1)
    phys = active.clamp_min(0).to(torch.int32)
    neg = torch.tensor(-1, dtype=torch.int32, device=slots.device)
    pages = torch.where(valid, torch.div(phys, page, rounding_mode="floor"), neg)
    rows = torch.where(valid, phys.remainder(page), neg)
    flat = torch.where(
        valid, active, torch.tensor(-1, dtype=torch.int64, device=slots.device)
    ).to(torch.int64)
    return pages, rows, flat


def _run(slots, positions, qsl, num_reqs, num_actual_tokens, page, ratio, skip):
    num_tokens = slots.numel()
    coords = torch.full((num_tokens, 2), -7, dtype=torch.int32, device=slots.device)
    flat = torch.full((num_tokens,), -7, dtype=torch.int64, device=slots.device)
    build_packed_cache_slot_mapping(
        slots,
        positions,
        qsl,
        num_tokens,
        num_reqs,
        num_actual_tokens,
        page,
        ratio,
        skip_update=bool(skip),
        coordinates_output=coords,
        flat_output=flat,
    )
    torch.npu.synchronize()
    return coords[:, 0].clone(), coords[:, 1].clone(), flat.clone()


def _fixture(tokens, num_reqs, dtype=torch.int32, sentinel_every=7, seed=0):
    torch.manual_seed(seed)
    slots = torch.randint(0, 200000, (tokens,), dtype=dtype, device=DEVICE)
    if sentinel_every:
        slots[::sentinel_every] = -1
    positions = torch.randint(0, 1000, (tokens,), dtype=torch.int64, device=DEVICE)
    per = tokens // num_reqs
    qsl = torch.arange(0, tokens + 1, per, dtype=torch.int32, device=DEVICE)
    return slots, positions, qsl


# ``page``, ``ratio``, ``skip``, ``slots dtype``, ``num_actual_tokens``
MATCHED_EAGER = [
    (128, 1, 0, torch.int32, None),
    (64, 2, 0, torch.int32, None),
    (64, 2, 1, torch.int32, None),
    (96, 1, 0, torch.int32, None),    # non power-of-two page
    (100, 1, 0, torch.int32, None),   # non power-of-two page
    (96, 1, 0, torch.int64, None),    # 64-bit slots keep the wide-arithmetic path
    (64, 2, 0, torch.int64, None),
]


@pytest.mark.parametrize("page,ratio,skip,dtype,num_actual", MATCHED_EAGER)
@pytest.mark.parametrize("tokens,num_reqs", [(96, 16), (80, 16), (6, 6), (2048, 16)])
def test_matches_eager_chain(page, ratio, skip, dtype, num_actual, tokens, num_reqs):
    """Arithmetic-only rewrite: every supported page/ratio/dtype agrees."""
    slots, positions, qsl = _fixture(tokens, num_reqs, dtype)
    actual = tokens if num_actual is None else num_actual
    got = _run(slots, positions, qsl, num_reqs, actual, page, ratio, skip)
    want = _reference(slots, positions, qsl, num_reqs, actual, page, ratio, skip)
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@pytest.mark.parametrize("ratio,page", [(1, 128), (2, 64)])
def test_padded_tail_is_masked_for_ratio1(ratio, page):
    """Rows past ``num_actual_tokens`` never reach a packed cache plane.

    A full-graph replay keeps a fixed-size token carrier, so the tail of the
    slot mapping can still hold ids from an earlier step. The kernel masks them
    for *both* ratios; this pins that behaviour.
    """
    tokens, num_reqs = 96, 16
    slots, positions, qsl = _fixture(tokens, num_reqs)
    num_actual = 60
    pages, rows, flat = _run(slots, positions, qsl, num_reqs, num_actual, page, ratio, 0)
    assert torch.all(flat[num_actual:] == -1)
    assert torch.all(pages[num_actual:] == -1)
    assert torch.all(rows[num_actual:] == -1)
    if ratio == 1:
        # ...and the eager chain would have written them, which is the trap.
        eager_flat = _reference(
            slots, positions, qsl, num_reqs, num_actual, page, ratio, 0
        )[2]
        assert not torch.equal(eager_flat, flat)


def test_all_sentinel_writes_nothing():
    """An all-``-1`` mapping (no active KV slot) must be a clean no-op."""
    tokens, num_reqs = 96, 16
    slots, positions, qsl = _fixture(tokens, num_reqs, sentinel_every=1)
    assert torch.all(slots == -1)
    pages, rows, flat = _run(slots, positions, qsl, num_reqs, tokens, 128, 1, 0)
    assert torch.all(flat == -1)
    assert torch.all(pages == -1)
    assert torch.all(rows == -1)


def test_single_token_batch():
    """``tokens=1`` still launches (grid of one) and matches the eager chain."""
    slots, positions, qsl = _fixture(1, 1, sentinel_every=0)
    got = _run(slots, positions, qsl, 1, 1, 128, 1, 0)
    want = _reference(slots, positions, qsl, 1, 1, 128, 1, 0)
    for g, w in zip(got, want):
        assert torch.equal(g, w)


def test_rejects_bad_page_size_and_ratio():
    """Argument validation happens before any launch."""
    slots, positions, qsl = _fixture(96, 16)
    coords = torch.empty((96, 2), dtype=torch.int32, device=DEVICE)
    flat = torch.empty((96,), dtype=torch.int64, device=DEVICE)
    with pytest.raises(ValueError):
        build_packed_cache_slot_mapping(
            slots, positions, qsl, 96, 16, 96, 128, 3,
            coordinates_output=coords, flat_output=flat,
        )
    with pytest.raises(ValueError):
        build_packed_cache_slot_mapping(
            slots, positions, qsl, 96, 16, 96, 0, 1,
            coordinates_output=coords, flat_output=flat,
        )
