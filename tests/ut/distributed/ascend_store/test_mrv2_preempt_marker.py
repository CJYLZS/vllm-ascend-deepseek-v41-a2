#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""MRV2 preempted-request marker must be consumed by _process_new_request.

Upstream context (vllm-ascend #19693 / commit 6197fec5c):

MRV2 resumes preempted requests through ``scheduled_new_reqs`` carrying a FULL
block table. ``KVPoolScheduler`` rebuilds the tracker but, before the fix, left
``_preempted_req_ids`` set. The next RUNNING step then supplied a block delta,
took the preempted-request branch, replaced the full table and reset save
progress -- surfacing as ``ReqMeta GVA metadata does not cover requested block
range`` or, worse, associating the wrong blocks with pool keys (silent).

The fix is two lines in ``_process_new_request`` (gated on
``use_v2_model_runner``).
"""

import collections
import inspect
from types import SimpleNamespace

import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)

REQ_ID = "req-mrv2-preempt"


def _bare_scheduler(use_v2):
    """Minimal scheduler carrying only the state _process_new_request touches."""
    sched = KVPoolScheduler.__new__(KVPoolScheduler)
    sched.vllm_config = SimpleNamespace(use_v2_model_runner=use_v2)
    sched._preempted_req_ids = {REQ_ID}
    sched.load_specs = collections.OrderedDict()
    return sched


def _request():
    return SimpleNamespace(
        req_id=REQ_ID,
        num_computed_tokens=0,
        block_ids=[],
        prompt_token_ids=[],
    )


def _scheduler_output():
    return SimpleNamespace(num_scheduled_tokens={REQ_ID: 1})


def _drive(sched):
    """Call _process_new_request, tolerating failures past the marker fix.

    The fix sits in the first statements; anything deeper needs a full KV-cache
    config and pool worker. We only care whether the marker was consumed.
    """
    try:
        sched._process_new_request(_request(), _scheduler_output(), force_skip_save=False)
    except Exception:
        pass


@pytest.mark.parametrize("use_v2", [True, False])
def test_preempted_marker_consumed_only_on_mrv2(use_v2):
    sched = _bare_scheduler(use_v2=use_v2)
    assert REQ_ID in sched._preempted_req_ids, "precondition: marker starts set"

    _drive(sched)

    if use_v2:
        assert REQ_ID not in sched._preempted_req_ids, (
            "MRV2 must clear the preempted marker in _process_new_request; "
            "otherwise the next RUNNING delta rebuilds the tracker from a "
            "delta and can associate wrong blocks with pool keys"
        )
    else:
        assert REQ_ID in sched._preempted_req_ids, (
            "MRV1 resumes through cached requests and consumes the marker in "
            "_process_preempted_cached_request"
        )


def test_fix_is_gated_on_use_v2_model_runner():
    """An ungated discard would change MRV1 behaviour."""
    src = inspect.getsource(KVPoolScheduler._process_new_request)
    head = "\n".join(src.splitlines()[:20])
    assert "use_v2_model_runner" in head, (
        "_process_new_request should consult use_v2_model_runner before "
        "discarding the preempted marker (see upstream #19693)"
    )
    assert "_preempted_req_ids.discard" in head, (
        "_process_new_request should discard the preempted marker for MRV2"
    )
