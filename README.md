# vllm-ascend-deepseek-v41-a2

在 **8 × Ascend 910B3（A2）** 单机上服务
[`DeepSeek-V4.1-Flash-1017-w4a8-Ascend`](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend)，
基于官方 **vllm-ascend v0.30.0rc2**，默认启用 **Model Runner V2**。权重获取见 §1.1。

| 文件 | 说明 |
|---|---|
| `series.conf` | **补丁序列（唯一入口）**：按行声明启用哪些 patch，`#` 注释 |
| `patches/0001..0004` | 性能/功能优化（默认启用） |
| `patches/0101-*` | **KV offload 修复**（仅挂 `--kv-transfer-config` 时需要） |
| `tests/` | 本包自带回归用例（`test_mrv2_preempt_marker.py`） |
| `vllm.patch` | vllm 侧，1 文件 +76/−4 |
| `legacy/vllm-ascend.monolithic.patch` | **已废弃**：旧的单个大补丁，仅作考古/回退 |
| `start-server.sh` | 起服脚本（镜像 / 权重 / 端口全部由环境变量传入） |
| `README.md` | 本文件 |

基线 commit：`vllm-ascend v0.30.0rc2`（`8fde6d4e3`）、`vllm v0.30.0`（`ced6857af`）。
补丁只含本地改动，不含任何上游新提交。

### 0.1 为什么拆成 `patches/` + `series.conf`（2026-10-27 重构）

原来是一个 `vllm-ascend.patch`（8 文件 +411/−64）。拆分的三个理由：

1. **可分层**：`0xxx` 是纯优化、`1xxx` 是 KV offload。不挂池时不必承担 `1xxx`。
2. **可演进**：升级基线时按「上游 churn」逐个评估，而不是重做一整块。
   实测 `v0.30.0rc2 → main`（452 提交）的 churn：
   `dsa_v1.py` **11 提交/+623−159**（重灾区）、`dsa_v41.py` 5/+153−4、
   `model.py` 7/+46−1、`rope_dsv4.py` 1，
   而 `engram/layer.py`、`indexer.py`、`token_dispatcher.py` **各 0 提交 ⇒ 零冲突**。
3. **可核对**：拆开后每个 patch 一个主题，review 与回退都以主题为单位。

> 拆分经过等价性验证：按 `series.conf` 顺序应用后的源码树，
> 与旧单个 `.patch` 应用后的结果 `diff -r` **零差异**。

---

### 0.2 用补丁 / 加卸载 / 升级基线

```bash
# 普通起服（默认序列，不挂 KV 池）
bash start-server.sh

# 加 KV offload 外部池：一个开关，MetaService 容器 + conf + vllm 参数全自动
KV_OFFLOAD=1 bash start-server.sh
KV_OFFLOAD=1 KV_DRAM_GB=4 bash start-server.sh     # 池大小：4GB/卡 ⇒ 整机 32GB

# 临时不启用某个补丁：注释掉 series.conf 里那一行即可
#   （改完指纹会自动变，下次起服重建 tree/，不会再出现"改了没生效"）
```

> `KV_OFFLOAD` 默认 **0**，关掉时脚本行为与从前逐字一致（不建容器、不写 conf、不加参数）。

**升级基线（例如 0.31.0rc1）**：`tree/` 是**真 git 克隆**，用它做 rebase 比 `patch -p1` 精确：

```bash
cd tree/vllm-ascend
git checkout v0.31.0rc1          # 或对应 tag
git checkout -b a2/v0.31.0rc1
for p in $(grep -v '^#' ../series.conf | grep -v '^$'); do
  git apply --3way ../patches/$p || echo "冲突: $p"   # --3way 给出精确冲突点
done
# 解决后导出为新基线的 patch 集：
git format-patch v0.31.0rc1 -o ../patches-new/
```

按 §0.1 的 churn 判断优先级：
**先看 `0004-model-opts`**（碰 `model.py`，7 提交）与 **`0002-dsa-v41`**（5 提交），
`0003-dsa-v1-o-proj-2d` 冲突虽大但改动由 `VLLM_ASCEND_V41_O_PROJ_2D` 门控、默认关闭，
**可以延后**；`0004` 里那三个零 churn 文件（`engram/layer.py`、`indexer.py`、`token_dispatcher.py`）
可原样保留。

> ⚠ 不要因为「V2 默认走 `dsa_v41`」就删掉 `0003`：`dsa_v1.py` 在 V2 下**确实会被 import**
> （`worker/v2/attn_utils.py:64` 引用 `AscendDSAMetadataBuilder`）。

### 0.3 KV offload 补丁 0101 说明

上游 `vllm-ascend#19693`（commit `6197fec5c`）修的是 **MRV2 × KVPool** 的真实缺陷：

MRV2 通过 `scheduled_new_reqs` 带**完整 block 表**恢复被抢占的请求，但 `KVPoolScheduler`
重建 tracker 后没清 `_preempted_req_ids`。下一步 RUNNING 带着增量表进入抢占分支，
**用增量覆盖完整表**，导致 `ReqMeta GVA metadata does not cover requested block range`，
或把**错误的 block 关联到池 key**（静默算错，不崩溃）。

修复 2 行、门控在 `use_v2_model_runner` 下（MRV1 走 `_process_preempted_cached_request`，
不受影响）。本包附 `tests/ut/distributed/ascend_store/test_mrv2_preempt_marker.py` 覆盖它。

### 0.4 KV offload 起服手册（`KV_OFFLOAD=1`）

三件事缺一不可，脚本会全部自动处理；**缺任何一件，vllm 都会在 engine 初始化时
（起服约 8 分钟后）`assert res == 0` 死掉**，所以脚本把能前置的全部前置了。

| 项 | 脚本行为 |
|---|---|
| MetaService 容器 `<NAME>-mmc-meta` | 不存在则起；已存在则**先校验端口匹配**再复用（不重启，重启会丢池内容） |
| `mmc-local.conf` / `mmc-meta.conf` | 生成到 `runtime/<HOST>/<NAME>/kv/`，只读挂到 `/workspace/kvconf` |
| `-e MMC_LOCAL_CONFIG_PATH` | 指向挂载后的 conf |
| `--kv-transfer-config` | `AscendStoreConnector` + `kv_both` + `memcache` |
| 就绪闸门 | 等 `:<metrics>/metrics` 起来，60 s 不行就 `exit 6`（不浪费 9 min） |

**可调变量**（都给默认值）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `KV_DRAM_GB` | **2** | ★ **每 rank** 的 DRAM 池；8 卡 ⇒ 整机 16 GB |
| `KV_PROTOCOL` | `host_shm` | 见下，别改 |
| `KV_WORLD_SIZE` | 256 | 可登记 rank 上限，改了要重启 meta |
| `KV_META_PORT` / `KV_STORE_PORT` / `KV_METRICS_PORT` | 5100 / 6100 / **8100** | metrics 不能是 8000，会和 `--port` 撞 |

#### 四个写死在模板里的坑（各踩过一次起服）

1. **`protocol` 必须 `host_shm`**。`host_rdma` 的 extend 库在镜像是 **0 字节**
   （`LoadExtendLibrary for HOST RDMA failed: -6`）；`device_rdma` 单 rank 能过、
   **8 rank 必 segfault**（`RaSocketBatchConnect`）—— 别用单 rank 结论外推。
2. **`dram.size` 是每 rank**。写 `8GB` 以为是整机，实际要 64 GiB ⇒
   `halMemAlloc failed, ret:6`。
3. **不要 `docker run -e LD_LIBRARY_PATH=...`**。它会覆盖镜像的 CANN 路径 ⇒
   `ImportError: libhccl.so`，15 s 退出。本脚本不设这个变量。
4. **`metrics_url` 别用 8000**。镜像自带模板 `mmc-meta.conf` 就是 8000，直接 `cp`
   会跟 vllm 抢端口。

#### 怎么确认"真的生效了"（`WAIT=1` 已自动校验）

池是**静默失效**型：connector 名字打出来了但池没连上，服务照常起、只是永远不命中。
三条同时成立才算真生效：

```bash
# ① 8 个 rank 都建了 connector
grep -c 'Creating v1 connector with name: AscendStoreConnector' <log>     # 期望 = 8
# ② 8 个 rank 都向 meta 登记了容量（这条才是"池真接上了"）
grep -c 'RegisterBaseInfo.*register capacity' <log>                       # 期望 = 8
# ③ meta 侧容量对得上：KV_DRAM_GB × 8
curl -s http://127.0.0.1:8100/metrics | grep '^memcache_total_capacity_bytes'
```

命中效果则看指标（跑完一轮压测后再看）：

```bash
curl -s http://127.0.0.1:8100/metrics | grep -E \
  'memcache_(allocated_bytes|alloc_successes_total|batch_query_not_found_total|evict_operations_total)'
```

> ⚠ **池容量口径**：全量 KV 是 `29963 block × 1 MiB = 29.26 GiB/机`。
> 2 GB/rank（16 GB/机）≈ 全量的 55%；这个尺寸下单轮 8K 压测就会开始驱逐。
> 驱逐只丢块、不损坏正确性。

> ⚠ **正确性判据必须用首 token**。同 prompt 连发 4 次（全命中）会有 2 种输出，
> 差别仅在"但最后问"/"后问"这类同义抖动 —— 整段比对不可靠。

## 0. 实测数据

单机 8 × 910B3（A2），TP8/EP8。协议：8K prompt → 512 输出，reps=3，temperature=0。

### 与 V1 对比

| 并发 | 1 | 4 | 5 | 8 | 9 | 13 |
|---|---|---|---|---|---|---|
| V1 每步 (ms) | 46.33 | 55.40 | 57.46 | 60.21 | 62.51 | 67.80 |
| **V2 每步 (ms)** | **33.00** | **42.77** | **45.35** | **50.10** | **50.76** | **55.35** |
| 每步降幅 | −28.8% | −22.8% | −21.1% | −16.8% | −18.8% | −18.4% |
| V2 吞吐 (tok/s/流) | 120.0 | 70.6 | 58.1 | 56.3 | 51.5 | 48.7 |

每步时延可复现到 0.3%，是最硬的判据。吞吐受投机解码接受长度抽签影响，
单发波动 ±15%，只作参考。

### KV 容量

**3,642,093 tokens**（V1 是 3,633,276，差 0.24%，在 `block_size=128` 取整的浮动范围内）。

### 口径说明

V2 数据为本机实测；V1 一行为历史基准（同机同协议、不同次启动）。
两组之间的差距**同时包含**「V2 runner」与「V2 用不上 6 个 dspark 开关」两个因素，
不是纯 runner 对比。同一天同机背靠背测的 8K→256 那组给出 −20.3~−31.3%，量级一致。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 硬件 | 单机 8 × 910B3（A2），每卡 HBM 65536 MB（可用约 61 GB） |
| OS / 架构 | Linux `aarch64`（实测 openEuler 22.03 LTS-SP4；镜像本体是 Ubuntu 22.04） |
| 驱动 | 25.5.6，`/usr/local/Ascend/driver` 就位 |
| 镜像 | `vllm-ascend:v0.30.0rc2`。公开可用 `quay.io/ascend/vllm-ascend:v0.30.0rc2`（multi-arch，含 arm64） |
| 权重 | `DeepSeek-V4.1-Flash-1017-w4a8-Ascend`，目录下要有 `config.json`，见 §1.1 |
| 其他 | `docker`、`tar`；有 `git` 或 `patch` 之一（打补丁用） |

### 1.1 获取权重

ModelScope：**[chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend)**

```bash
pip install modelscope
modelscope download --model chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
    --local_dir /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

cd /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
sha256sum --check SHA256SUMS
cd engram_int8
bash reassemble_engram_weights.sh
```

> ⚠ **最后一步 `reassemble_engram_weights.sh` 不能省。** 仓库里两张 Engram 权重
> （`layers_1_...` / `layers_14_...`，各约 98 GB）是以 **6 片分片**上传的
> （`part-00-of-06` … `part-05-of-06`），运行时只认重组后的**单文件**
> `.safetensors`，不会自动读 `.part-*`。跳过重组直接起服会报：
>
> ```
> ValueError: Engram layer 1: the checkpoint index points at
> .../engram_int8/layers_1_engram_embed.weight.safetensors, which does not exist.
> ```
>
> 脚本会先校验分片清单连续性、逐片 SHA256，拼接后再校验完整表 SHA256，最后原子替换。
> 需要为两张还原表额外预留约 **184 GiB** 磁盘；已还原且校验正确的文件会被直接复用，
> 重复执行是安全的（幂等）。
>
> ```bash
> bash reassemble_engram_weights.sh --check-only     # 只校验，报告是否需要重组
> bash reassemble_engram_weights.sh --delete-parts  # 重组成功后删掉分片，省约 184 GiB
> ```
>
> 删掉分片后根目录 `SHA256SUMS` 里对应条目会自然缺失，此时改用
> `cd engram_int8 && sha256sum --check WEIGHTS.sha256` 校验完整表。

主要构成：

| 部分 | 大小 | 说明 |
|---|---|---|
| `quant_model_weights-*-of-00080.safetensors` | 292 GB | 80 个分片，w4a8 量化主权重 |
| `engram_int8/*.part-00-of-06` … `part-05-of-06` | 196 GB | 两张 Engram 表的 6 片分片（需重组） |
| `engram_int8/layers_{1,14}_engram_embed.weight.safetensors` | 196 GB | 重组后的完整表（脚本生成） |
| `engram_extra.safetensors` | 0.6 GB | engram 附加权重 |
| `mtpq-*-of-00004.safetensors` | 10 GB | MTP（dspark 投机解码）权重 |
| `quant_model_description.json` | 11 MB | 量化描述，加载时读 |

磁盘按三个阶段算：

| 阶段 | 占用 | 说明 |
|---|---|---|
| 刚下载完 | **约 500 GB** | 只有分片，无完整表 |
| 重组后（分片保留） | **约 695 GB** | 分片与完整表并存，建议按此预留 |
| 重组后 `--delete-parts` | 约 500 GB | 删分片回收 196 GB |

> **`engram_int8/` 不能省**。它是 engram 层的 int8 权重，占整个权重的绝大部分；
> 删掉或只下一部分，加载阶段就会报缺文件。分片文件必须整组齐全。

权重目录只需**只读**挂载（脚本用 `:ro`），可以放在 NFS / 共享存储上多机复用；
但路径一旦确定就别改 —— 编译缓存的键含模型路径，换路径等于全量冷启动。

---

## 2. 快速开始

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc2
export MODEL=/path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

bash start-server.sh                    # 起服（首次会自动准备源码树并打补丁）
FORCE=1 WAIT=1 bash start-server.sh     # 重启 + 等 READY + 自检
DRY_RUN=1 bash start-server.sh          # 只打印 docker 命令，不起容器
```

起服到 READY 约 **9~11 min**（目标模型 32 张图 × 8 卡），**首次**再加编译缓存冷启动。
停服务 `docker stop dsv41`；健康检查：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
```

服务名 `DeepSeek-V4.1-Flash`，端口 8000。

---

## 3. 源码树是怎么来的（关键，别跳）

`start-server.sh` 首次运行会做这件事，**不需要手工操作**，但需要理解为什么：

```bash
# 从镜像里取出 stock 源码树（排除编译中间产物，约 0.6 GB）
docker run --rm --entrypoint tar "$IMAGE" -C /vllm-workspace -cf - \
    --exclude='vllm-ascend/csrc/build' --exclude='vllm-ascend/csrc/build_cache' \
    --exclude='vllm-ascend/csrc/build_out' --exclude='vllm-ascend/csrc/output' \
    --exclude='vllm/rust/target' vllm-ascend vllm | tar -xf - -C "$SRC_DIR"

# 再按 series.conf 顺序应用补丁
for pf in $(grep -v '^#' series.conf | grep -v '^$'); do
    git -C "$SRC_DIR/vllm-ascend" apply "patches/$pf"
done
git -C "$SRC_DIR/vllm" apply vllm.patch
```

> 是否重建树由 **序列指纹**（`series.conf` + 各 patch 的 md5）决定，
> 不是"标记文件存在就跳过"。改任何补丁都会自动重建 —— 旧实现只看文件存在，
> 曾导致 v2 补丁已裁到 796 行、`tree/` 里仍是 v1 的 941 行改动却照用不误。

**为什么不能 `git clone` 一份再打补丁**：本包的改动全在 Python 侧，但运行必需的构建产物
不在 git 里（被 `.gitignore` 排除）：

```
vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so    # C 扩展，缺了直接 ImportError
vllm_ascend/libvllm_ascend_kernels.so
vllm_ascend/lib/、vllm_ascend/_cann_ops_custom/（44 MB）、vllm_ascend/include/
vllm/vllm/_rust_tool_parser.abi3.so、vllm/vllm/vllm-rs（54 MB）
```

从**镜像自带的树**里取出来，这些产物天然配对，不用重新编译（否则要编 C 扩展 +
48 个 CANN 自定义算子 + rust parser）。补丁本身不改 C++/rust 源码，只改 `.py`，
所以复用镜像里的 `.so` 是安全的。

已准备好的树可以复用：`SRC_DIR` 下有 `.patches-applied` 标记就跳过准备步骤。
想重来就删掉那个标记（或整个 `SRC_DIR`）。

> 如果你已经有自己维护的源码树和构建产物，设 `SKIP_PREPARE=1` 并让 `SRC_DIR` 指向它，
> 脚本只做产物存在性 + gate 定义检查。

---

## 4. 参数说明（默认值 = 实测最优）

可用同名环境变量覆盖任意一项。

| 参数 | 默认 | 依据 |
|---|---|---|
| `TP` | 8 | 单机 8 卡全用，`--enable-expert-parallel` |
| `MAXLEN` | 1048576 | 长上下文；改它会让编译缓存失效（进 `SchedulerConfig.compute_hash()`） |
| `BAT` | 4096 | 4096 vs 16384 吞吐差异小于运行噪声，但 KV 容量是 3.6M vs 1.3M token ⇒ 按容量选 |
| `MAX_SEQS` | 32 | 同上，进 compute_hash |
| `GPU_UTIL` | 0.92 | 留余量给 plog；0.90 也能跑 |
| `BLOCK` | 128 | prefix cache 粒度，命中量按 128 取整 |
| `SPEC_TOKENS` | 5 | dspark 的 MTP 按「一块 5 token」训练，K=10 实测更差（每步 +23.4%，接受 token 只 +20%） |
| `CUDAGRAPH_SIZES` | 34 桶 | 上游 27 桶 + 补 7 个 `6N`。draft 一步吃 `num_reqs×(1+SPEC_TOKENS)` token，缺 6N 时 ACL graph 补位把 draft 打回 eager，见 §0 |
| `SRC_MOUNT_MODE` | ro | 想改代码调试就设 `rw` |

`--enable-prompt-tokens-details` 已内置。上游默认关闭，导致 `usage.prompt_tokens_details`
恒为 `null`，**prefix cache 命中多少 token 在 API 里完全看不到**。开启后每请求返回
`{"cached_tokens":…, "created_cache_tokens":…}`。纯观测性开关，不影响调度 / 数值 / 图结构。
不需要就删脚本里那一行。

---

## 5. 可选：PGO 版 libpython

上游镜像里的 CPython 没开 PGO。换成 PGO+LTO 重编的 `libpython3.12.so.1.0`，
实测每步 decode 时延下降 **3.2~5.1%（六档均值 −4.27%）**，纯 Python 微基准快 1.59~1.61×。

```bash
PGO_SO=/path/to/libpython3.12.so.1.0 bash start-server.sh
```

脚本会把它挂到 `/usr/local/python3.12.13/lib/libpython3.12.so.1.0`。
镜像原版 md5 是 `0709a2b39e034d403f3d72788ba12beb`，回退直接不设 `PGO_SO`。

⚠ 这个 `.so` 必须**在目标机上编译**（构建指纹含 CPU 型号、gcc、glibc、源码 sha）。
本仓库不含该产物。不装不影响其余优化。

> PGO 的收益只能看**每步时延**。吞吐会被投机解码接受长度 `A_spec` 的抽签噪声盖住
> （实测 CV 7.7~9.9%，大于 4.3% 的信号），跨进程比吞吐会得出相反结论。

---

## 6. 静默失效：6 个开关必须都传

下面 6 个 env **默认值全是 0**，不传时服务照常起、日志不报错，但收益全丢。

```
VLLM_ASCEND_V41_TRITON_SLOT_MAPPING     单 launch Triton 槽位映射
VLLM_ASCEND_ROPE_INDEX_SELECT           cos/sin 走 index_select（省 BroadcastTo+Cast）
VLLM_ASCEND_V41_MOE_MASK_RANGE          连续 expert 区间比较（去 2× Cast/层）
VLLM_ASCEND_V41_O_PROJ_2D               wo_a 退化 batch matmul → 2D matmul
VLLM_ASCEND_V41_QLI_NO_CANDIDATE        每层走 mode 3，关掉两级候选机制
VLLM_ASCEND_V41_ENGRAM_WKV_TP           engram wkv 按输出维分片（每卡省 550 MB → 换 KV 容量）
```

**全部在 import 期读取，且在捕获图里 ⇒ 只能重启切换，热改 env 无效。**

V1 分支里有 12 个，本分支只留 6 个。另外 6 个 `VLLM_ASCEND_DSPARK_*` 只在
`spec_decode/dspark_proposer.py` 和 `llm_base_proposer.py` 里被读，而这两个文件属于
V1 的 proposer 路径。V2 走 `worker/v2/spec_decode/dspark/speculator.py`（继承上游
`DSparkSpeculator` + `DFlashAclGraphManager`），根本不 import 它们。V1 里那 6 个开关
换来的「draft 进 ACLGraph」，在 V2 由 `query_cudagraph_manager` 原生提供。
传了不报错也不生效，所以本分支既不定义也不传。

`start-server.sh` 因此做了两道防护：

1. 起服前扫源码树的 `envs.py`，确认每个 gate 都有定义，缺了直接 `exit 3` 列出缺项
2. `WAIT=1` 时硬校验真的走了 V2，判据见 §7

---

## 7. 部署后验证

```bash
# 1) 服务活着
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 期望 200

# 2) 参数真的生效了（不是"起成功了"就算）
docker inspect dsv41 --format '{{json .Config.Cmd}}' | tr ',' '\n' | grep -E 'speculative|enable-prompt'
docker inspect dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep VLLM_ASCEND

# 3) 确认真的走了 V2（V1 的判据在 V2 下恒为 0，不能拿来用）
docker logs dsv41 2>&1 | grep -c 'npu model runner v2 is in developing'      # 期望 8
docker logs dsv41 2>&1 | grep -c '\[dspark-graph-capture\]'                   # 期望 0
#    目标图应在日志里看到 "Capturing CUDA graphs (FULL): 100%|...| 32/32"
#    "ACL graph replay is active for the V2 target model" 要点：它在**首个请求之后**
#    才打印（logger.info_once），起服完成时看不到是正常的

# 4) prefix cache 观测可用（同一个 prompt 发两次，看 cached_tokens 应为 0 → 非 0）
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"DeepSeek-V4.1-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c "print('hello world '*200)")\"}],\"max_tokens\":8}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

第 4 步注意：短 prompt（<128 token）`cached_tokens` 恒为 0，这是 `block_size=128` 的
取整粒度，**不是没生效**。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FATAL: 缺构建产物 ...vllm_ascend_C...so` | `SRC_DIR` 是空的 git 树，不是从镜像里取的。删掉 `SRC_DIR` 重跑 |
| `FATAL: 这些 gate 传了 =1，但 envs.py 里没有定义` | 补丁没打成功，或 `SRC_DIR` 指向未打补丁的树。检查 `.patches-applied` |
| 起服到了但明显慢，日志没报错 | 6 个开关少了几个。用 §7 第 2 步核对 |
| `ImportError: libascend_hal.so` / `Failed to load the backend extension: torch_npu` | 在裸 `docker run` 里 import 了 vllm；必须带全部 `--device` 和驱动挂载 |
| 起不来但没留下有用日志 | 脚本保留 log、plog、npulog，`docker stop` 后仍可查 `$LOG_DIR/logs/` |
| KV 容量上报每次差 ±128 | 固定 HBM 下按 `block_size=128` 取整分配，不是参数变化 |
| `vllm serve --help` 找不到 `--enable-prompt-tokens-details` | 顶层只列 Config Groups，要 `--help=Frontend` |

logs / plog / npulog / 编译缓存都落在 `$LOG_DIR`（默认 `./runtime/<hostname>/<NAME>/`），
按主机隔离。多台机共用同一份脚本目录时不会互相覆盖。

---

## 9. 兼容性提醒

- **不要用 `vllm_ascend/_version.py` 判断实际跑的代码**。它由构建期生成，本补丁没有
  重新生成它，所以仍会报 `0.30.0rc2` / `g8fde6d4e3` —— 落后 4 个提交。要看真实状态就
  `git -C $SRC_DIR/vllm-ascend log --oneline -1`。
- 补丁针对 `v0.30.0rc2` 原样应用。若你的镜像里 `/vllm-workspace` 的树被改过，
  `git apply` 会失败 —— 脚本立刻报错，不会静默。
- 只验证过 8 卡 TP8/EP8 单机。改 `TP` 未测。
- **Model Runner V2 在 vllm-ascend 侧标注为 `[Experimental]`**（见
  `vllm_ascend/worker/v2/README.md`，仍列 5 条未补 gap）。官方 DeepSeek-V4 + dspark
  的 e2e 用例覆盖的是 A3，A2 无官方覆盖 —— 本包的数据来自本机实测，不是持续的 CI 覆盖。
- V2 的图捕获规模与 V1 不同：目标模型 **32 张**、draft **24 张**（V1 是 draft 32 桶 ×
  8 卡 = 256 条）。draft 的 24 = 桶表按 `ceil(s/5)` 去重后，被
  `max_num_seqs × num_query_per_req(5) = 160` 截掉超出部分的结果。
- dspark 的 `num_speculative_tokens` 只能是 `n_predict=5` 的整数倍，`SPEC_TOKENS=10`
  实测更差（每步 +23.4%，接受 token 只 +20%）。
- 跨会话比较 `A_spec` 会得出相反结论：每步成本能复现到 0.3%，`A_spec` 只能复现到 8~16%。
  要比 decode 快慢，用同一次运行内的 `A_spec×1000/ITL`，或固定 prompt 多发均值。

---

## 10. 改动概要

本分支相对 V1 分支只保留与 V2 共用的改动。补丁 8 文件 +411/−64，按主题分：

| 主题 | 内容 |
|---|---|
| 主机路径去重 | 配置常量按 builder 记忆化、标量哨兵换缓存张量（一个 `where(…, 标量)` 在 NPU 上等于 5 个 host 算子）、槽位链在 KV 组内复用 |
| `where(out=)` | 原地写回，`copy_`/`empty_tensor` 各 −44 次/步 |
| 槽位映射合 kernel | 21 条槽位链（每链 106 个 host 算子，只算 6 个 token）换成单 launch Triton 内核 |
| 模型侧 | mask-range 区间比较；wo_a 2D；QLI 关候选；rope index_select；engram wkv TP 分片 |
| prompt 编码 | 空 developer 消息、控制 token 注入两个缺陷（在 `vllm.patch`） |

**已从本分支移除**（属 V1 专属，V2 不经过这些代码）：

| 移除项 | 原因 |
|---|---|
| `spec_decode/dspark_proposer.py` 全部改动 | V2 用 `worker/v2/spec_decode/dspark/speculator.py` |
| `spec_decode/llm_base_proposer.py` 全部改动 | 同上 |
| 6 个 `VLLM_ASCEND_DSPARK_*` 的 envs 定义 | 只有上面两个文件会读 |
| 6N 桶表说明（`CUDAGRAPH_SIZES` 仍保留同表） | V2 的 graph 匹配允许大图服务小批，不依赖精确命中 6N |

> V2 的 runner 能力全部来自镜像自带的 `worker/v2/`，**本补丁不修改 `worker/v2`
> 任何一行**。但 V2 **并非零补丁**：§6 那 6 个 gate 在 pristine 镜像里都没有定义，
> 全靠本补丁提供。
