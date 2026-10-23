# vllm-ascend-deepseek-v41-a2

在 **8 × Ascend 910B3（A2）** 单机上服务
[`DeepSeek-V4.1-Flash-1017-w4a8-Ascend`](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend)，
基于官方 **vllm-ascend v0.30.0rc2**，叠加 5 个提交的调优改动。权重获取见 §1.1。

| 文件 | 说明 |
|---|---|
| `vllm-ascend.patch` | vllm-ascend 侧，14 文件 +1244/−80 |
| `vllm.patch` | vllm 侧，1 文件 +76/−4 |
| `start-server.sh` | 起服脚本（镜像 / 权重 / 端口全部由环境变量传入） |
| `README.md` | 本文件 |

基线 commit：`vllm-ascend v0.30.0rc2`（`8fde6d4e3`）、`vllm v0.30.0`（`ced6857af`）。
两个补丁只含本地改动，不含任何上游新提交。

---

## 0. 实测数据

单机 8 × 910B3（A2），TP8/EP8。协议：8K prompt → 512 输出，reps=3，temperature=0。

### KV 容量

**3,633,276 tokens**（同配置重起在 ±128 内浮动，`block_size=128` 取整）。

### 吞吐（每流 tok/s）

| 并发 | 1 | 4 | 5 | 8 | 9 | 13 |
|---|---|---|---|---|---|---|
| tok/s | 74.6 | 55.2 | 47.0 | 42.3 | 42.6 | 41.2 |

### 每步 decode 时延（ms）

| 并发 | 1 | 4 | 5 | 8 | 9 | 13 |
|---|---|---|---|---|---|---|
| ms | 46.33 | 55.40 | 57.46 | 60.21 | 62.51 | 67.80 |

吞吐受投机解码接受长度抽签影响，单发波动约 ±15%；时延可复现到 0.3%。

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

起服到 READY 约 **9~11 min**（256 条图捕获 × 8 卡），**首次**再加编译缓存冷启动。
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

# 再打两个补丁
git -C "$SRC_DIR/vllm-ascend" apply vllm-ascend.patch
git -C "$SRC_DIR/vllm"        apply vllm.patch
```

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

## 6. 静默失效：12 个开关必须都传

这是本包最容易踩的坑。下面 12 个 env **默认值全是 0**，不传时：

- 服务**照常起、照常可用、日志不报错**
- 收益**全部丢失**

```
VLLM_ASCEND_DSPARK_DRAFT_GRAPH              draft 进 ACLGraph
VLLM_ASCEND_DSPARK_GRAPH_CAPTURE_METADATA
VLLM_ASCEND_DSPARK_CAPTURE_DISPATCH
VLLM_ASCEND_DSPARK_CAPTURE_VALUE_FIX
VLLM_ASCEND_DSPARK_CAPTURE_PAD_SLOTS
VLLM_ASCEND_DSPARK_DISPATCH_QUERY_LEN_FIX   c≥7 时 SparseFlashMlaMetadata 尺寸校验不 FATAL
VLLM_ASCEND_V41_TRITON_SLOT_MAPPING         单 launch Triton 槽位映射（每步 −2.2 ms）
VLLM_ASCEND_ROPE_INDEX_SELECT               cos/sin 走 index_select（省 BroadcastTo+Cast）
VLLM_ASCEND_V41_MOE_MASK_RANGE              连续 expert 区间比较（去 2× Cast/层）
VLLM_ASCEND_V41_O_PROJ_2D                   wo_a 退化 batch matmul → 2D matmul
VLLM_ASCEND_V41_QLI_NO_CANDIDATE            每层走 mode 3，关掉两级候选机制
VLLM_ASCEND_V41_ENGRAM_WKV_TP               engram wkv 按输出维分片（每卡省 550 MB → 换 KV 容量）
```

**全部在 import 期读取，且在捕获图里 ⇒ 只能重启切换，热改 env 无效。**

`start-server.sh` 因此做了两道防护：

1. 起服前扫 `$SRC_DIR/vllm-ascend/vllm_ascend/envs.py`，确认每个 gate 都有定义，缺了直接
   `exit 3` 并列出缺项 —— 不让你白等 9 分钟
2. `WAIT=1` 时硬校验 draft **真的**进图（`runtime_mode=FULL` 应为 8 条、
   `[dspark-graph-capture]` 应为 256 条），不对就 `exit 4`

draft 桶数怎么算（容易算错）：**不是**桶表长度 × 8，而是桶表经 `ceil(s/(1+SPEC_TOKENS))`
去重后的个数 × 8 卡。34 桶 → 32 → 256 条；去掉那 7 个 6N 补位（回到上游 27 桶）→ 25 → 200 条。

---

## 7. 部署后验证

```bash
# 1) 服务活着
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 期望 200

# 2) 参数真的生效了（不是"起成功了"就算）
docker inspect dsv41 --format '{{json .Config.Cmd}}' | tr ',' '\n' | grep -E 'speculative|enable-prompt'
docker inspect dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep VLLM_ASCEND

# 3) prefix cache 观测可用（同一个 prompt 发两次，看 cached_tokens 应为 0 → 非 0）
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"DeepSeek-V4.1-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"$(python3 -c "print('hello world '*200)")\"}],\"max_tokens\":8}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

第 3 步注意：短 prompt（<128 token）`cached_tokens` 恒为 0，这是 `block_size=128` 的
取整粒度，**不是没生效**。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FATAL: 缺构建产物 ...vllm_ascend_C...so` | `SRC_DIR` 是空的 git 树，不是从镜像里取的。删掉 `SRC_DIR` 重跑 |
| `FATAL: 这些 gate 传了 =1，但 envs.py 里没有定义` | 补丁没打成功，或 `SRC_DIR` 指向未打补丁的树。检查 `.patches-applied` |
| 起服到了但明显慢，日志没报错 | 12 个开关少了几个。用 §7 第 2 步核对 |
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
- dspark 的 `num_speculative_tokens` 只能是 `n_predict=5` 的整数倍，`SPEC_TOKENS=10`
  实测更差（每步 +23.4%，接受 token 只 +20%）。
- 跨会话比较 `A_spec` 会得出相反结论：每步成本能复现到 0.3%，`A_spec` 只能复现到 8~16%。
  要比 decode 快慢，用同一次运行内的 `A_spec×1000/ITL`，或固定 prompt 多发均值。

---

## 10. 改动概要

本包在 rc2 之上叠了 5 个提交，按主题分：

| 主题 | 内容 |
|---|---|
| dspark draft 进图 | 修 `SparseFlashMlaMetadata` 尺寸校验导致的 c≥7 崩溃；5 个 `CAPTURE_*` 开关 |
| 主机路径去重 | 配置常量按 builder 记忆化、标量哨兵换缓存张量（一个 `where(…, 标量)` 在 NPU 上等于 5 个 host 算子）、槽位链在 KV 组内复用 |
| `where(out=)` | 原地写回，`copy_`/`empty_tensor` 各 −44 次/步 |
| 槽位映射合 kernel | 21 条槽位链（每链 106 个 host 算子，只算 6 个 token）换成单 launch Triton 内核 |
| 模型侧 | 6N 桶表；mask-range 区间比较；wo_a 2D；QLI 关候选；rope index_select；engram wkv TP 分片 |
| prompt 编码 | 空 developer 消息、控制 token 注入两个缺陷 |
