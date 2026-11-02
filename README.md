# vllm-ascend-deepseek-v41-a2

在 **8 × Ascend 910B3（A2）** 单机上服务
[`DeepSeek-V4.1-Flash-1017-w4a8-Ascend`](https://www.modelscope.cn/models/chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend)，
基于官方 **vllm-ascend v0.30.0**，默认启用 **Model Runner V2**。

基线：`vllm-ascend v0.30.0`（`c09d3e20b`）、`vllm v0.30.0`（`ced6857`）、
CANN `9.2.1`、python `3.12.16`。**基座身份的唯一来源是本目录的 `base.conf`。**

> ### ⚠ 一个分支 = 一个基座
>
> | 分支 | 基座镜像 | 上游 commit | CANN | python | PGO libpython |
> |---|---|---|---|---|---|
> | **`v0300`（本分支）** | **`v0.30.0`** | **`c09d3e20b`** | **9.2.1** | **3.12.16** | **3.12.16 官方预编 PGO+LTO** |
> | `main` / `v1` / `v2` | `v0.30.0rc2` | `8fde6d4e3` | 9.2.0 | 3.12.13 | 3.12.13 本机自编 |
>
> 源码树是运行期产物（**不在本仓库里**，起服时从镜像现取）。各基座的树与 PGO 产物
> 按基座分家、互不覆盖，可随时 A/B 切换。
> 换基座 = 开新分支；`base.conf` 决定用哪个镜像、哪棵树、哪份 PGO。

## 实测数据

单机 8 × 910B3（A2），TP8/EP8，temperature=0。测的是**默认档**
（V2 runner + 6 个 gate + 共享专家多流 + PGO + `HCCL_OP_EXPANSION_MODE=AIV`
+ `0005` 槽位映射标量化，挂 2 GB/rank KV 外部池）。

> 每行都标注所属基座：**`v0.30.0` = 本分支（`v0300`）**，`v0.30.0rc2` = `main` 分支。
> 两套协议（512 / 1024）不可跨表比。判据只用**每步 `itl_ms`**，
> 不要用 `wall_ms_per_engine_output` —— 后者含 prefill 与 drain，会得出相反结论。

每步时延取 5 次重复的中位；吞吐取同一组重复中的最高值，并附该次的 `A_spec`。
`A_spec` 是不可控噪声（同档各次可差 2 倍，如 `c=1` 512 那档 1.46~4.34），
**只对每步时延下结论**；跨基座/跨会话比 `A_spec` 会得出相反结论。

**8K prompt → 512 输出**（每步 = `itl_ms`）：

| 并发 | 1 | 4 | 5 | 8 | 9 | 13 |
|---|---|---|---|---|---|---|
| **每步 (ms)** — `v0.30.0` | **27.31** | **36.01** | **38.07** | **41.31** | **43.73** | **48.99** |
| 每步 (ms) — `v0.30.0rc2` | 27.13 | 36.10 | 38.24 | 41.44 | 43.84 | 49.33 |
| 吞吐 (tok/s/流) — `v0.30.0` | 141.9 | 76.8 | 77.8 | 69.6 | 64.8 | 55.0 |
| A_spec — `v0.30.0` | 4.34 | 3.40 | 3.69 | 3.69 | 3.52 | 3.41 |

**8K prompt → 1024 输出**（与上表是两套协议，不可混比），最大并发 16：

| 并发 | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| **每步 (ms)** — `v0.30.0` | **27.27** | **35.56** | **41.57** | **51.72** |
| 每步 (ms) — `v0.30.0rc2` | 27.47 | 36.28 | 41.49 | 50.44 |
| 吞吐 (tok/s/流) — `v0.30.0` | 158.0 | 85.4 | 81.2 | 61.3 |
| 聚合吞吐 (tok/s) — `v0.30.0` | 158.0 | 341.6 | 649.5 | 980.8 |
| A_spec — `v0.30.0` | 4.59 | 3.64 | 4.22 | 3.93 |

**KV 容量**：3,628,926 tokens（`v0.30.0`）／3,629,044（`v0.30.0rc2`）。
同一配置重启在 ±128 内浮动（按 `block_size=128` 取整分配），不是参数变化。

> **两个基座每步时延无差异**：8K→512 平均 −0.22%（范围 −0.7%~+0.7%），
> 8K→1K 平均 ±0.00%（范围 −2.0%~+2.5%），方向在协议间不一致 ⇒ 噪声。
> 换基座是维护性升级，不是性能升级。

## 1. 前置条件

| 项 | 要求 |
|---|---|
| 硬件 | 单机 8 × 910B3（A2），每卡 HBM 65536 MB |
| OS / 架构 | Linux `aarch64` |
| 驱动 | 25.5.6 |
| 镜像 | **`vllm-ascend:v0.30.0`**（本分支；`main` 分支是 `v0.30.0rc2`）—— 取值见 `base.conf` |
| 权重 | 见 §1.1 |
| 其他 | `docker`、`tar`，`git` 或 `patch` 之一 |

### 1.1 获取权重

```bash
pip install modelscope
modelscope download --model chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend \
    --local_dir /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

cd /path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend
sha256sum --check SHA256SUMS
cd engram_int8 && bash reassemble_engram_weights.sh
```

> ⚠ **`reassemble_engram_weights.sh` 不能省**。两张 Engram 权重以 6 片分片上传，
> 运行时只认重组后的单文件 `.safetensors`。跳过会报
> `Engram layer 1: ... does not exist`。需额外约 **184 GiB** 磁盘，
> 重复执行是安全的；`--check-only` 只校验，`--delete-parts` 重组后删分片。

磁盘按三个阶段算：刚下载完 **约 500 GB** / 重组后（保留分片）**约 695 GB** /
`--delete-parts` 后 **约 500 GB**。

> `engram_int8/` 不能省，它占整个权重的绝大部分。权重只需只读挂载，可放 NFS 多机复用；
> **路径一旦确定别改** —— 编译缓存的键含模型路径，换路径等于全量冷启动。

---

## 2. 快速开始

```bash
# 镜像不必显式给：默认取 base.conf 的 BASE_IMAGE（本分支 = quay.nju.edu.cn/ascend/vllm-ascend:v0.30.0）
export MODEL=/path/to/DeepSeek-V4.1-Flash-1017-w4a8-Ascend

bash start-server.sh                      # 起服（首次自动准备源码树并打补丁）
FORCE=1 WAIT=1 bash start-server.sh       # 重启 + 等 READY + 自检
DRY_RUN=1 bash start-server.sh            # 只打印 docker 命令，不起容器

# 可选：挂 KV 外部池（§6）
KV_OFFLOAD=1 bash start-server.sh

# 可选：关掉共享专家多流（默认开，见 §4 参数表）
MULTISTREAM_SHARED_EXPERT=0 bash start-server.sh
```

起服到 READY 约 **9~11 min**，首次再加编译缓存冷启动。服务名 `DeepSeek-V4.1-Flash`，
端口 8000，容器名 `dsv41`（`NAME` 可覆盖）。

```bash
docker stop dsv41
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health
```

---

## 3. 源码树怎么来的

首次运行会自动做，不用手工操作：

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

为什么不 `git clone` 一份再打补丁：运行必需的构建产物不在 git 里（被 `.gitignore` 排除），
且必须与镜像配对 ——

```
vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so   # C 扩展，缺了直接 ImportError
vllm_ascend/libvllm_ascend_kernels.so
vllm_ascend/lib/、_cann_ops_custom/、include/
vllm/vllm/_rust_tool_parser.abi3.so、vllm/vllm/vllm-rs
```

补丁只改 `.py`，不改 C++/rust，所以复用镜像里的 `.so` 是安全的。

是否重建树由**序列指纹**（`series.conf` + 各 patch 的 md5）决定。改任何补丁都会自动重建；
想强制重来就删 `$SRC_DIR/.patches-applied`。已有自己的树和构建产物时设 `SKIP_PREPARE=1`。

---

## 4. 参数

可用同名环境变量覆盖任意一项。

| 参数 | 默认 | 说明 |
|---|---|---|
| `TP` | 8 | 单机 8 卡全用，`--enable-expert-parallel` |
| `MAXLEN` | 1048576 | 改它会让编译缓存失效（进 `SchedulerConfig.compute_hash()`） |
| `BAT` | 4096 | 4096 vs 16384 吞吐差异小于噪声，但 KV 容量是 3.6M vs 1.3M token ⇒ 按容量选 |
| `MAX_SEQS` | 32 | 同上，进 compute_hash |
| `GPU_UTIL` | 0.92 | 留余量给 plog |
| `BLOCK` | 128 | prefix cache 粒度，命中量按 128 取整 |
| `SPEC_TOKENS` | 5 | dspark MTP 按「一块 5 token」训练，K=10 实测更差 |
| `CUDAGRAPH_SIZES` | 34 桶 | 上游 27 桶 + 补 7 个 `6N`，缺 6N 时 draft 会被打回 eager |
| `SRC_MOUNT_MODE` | ro | 要改代码调试就设 `rw` |
| `PGO_SO` | 空 | PGO 版 libpython，见 §5 |
| `KV_OFFLOAD` | 0 | 挂 KV 外部池，见 §6 |
| `MULTISTREAM_SHARED_EXPERT` | 1 | 共享专家多流。上游镜像自带、**无需补丁**；实测每步 −1.3~−4.6%。生效自证 = 日志 `shared_experts.py:180` 的 `Shared expert split computation validation passed`。与 fused MC2 互斥 |
| `PASSTHRU_ENVS_STR` | 6 gate + AIV | 覆盖它会**整串替换**，6 个 gate 和 `HCCL_OP_EXPANSION_MODE=AIV` 一起丢（静默）。只加东西时请在自己那份里把默认串抄全，见 §7 |

`--enable-prompt-tokens-details` 已内置。上游默认关闭，导致 `usage.prompt_tokens_details`
恒为 `null`，prefix cache 命中量在 API 里看不到。开启后返回
`{"cached_tokens":…, "created_cache_tokens":…}`，纯观测性，不影响调度/数值/图结构。

---

## 5. 可选：PGO 版 libpython

默认已开。产物路径取自 `base.conf` 的 `BASE_PGO_SO_REL`，**挂载落点由
`base.sh` 的 `base_pgo_dest` 从镜像里探测**（起 0.7 s 一次性容器读 `sysconfig.LIBDIR`）。

> ⚠ **PGO 产物不在本仓库里**（编译产物，按 python 版本分家、放在仓库外），需自备。
> `BASE_PGO_SO_REL` 用相对路径指过去，指到不存在的文件时起服直接 FATAL（`exit 1`）。

```bash
# 默认：用 base.conf 里 BASE_PGO_SO_REL 指的那份（本分支 = 3.12.16 官方预编 PGO+LTO）
# 覆盖：PGO_SO=/path/to/libpython3.12.so.1.0 bash start-server.sh
# 关掉：PGO=0 bash start-server.sh
```

| 基座 | PGO 产物来源 |
|---|---|
| `v0.30.0rc2`（main） | 本机自编（python 3.12.13），md5 `68cfbe78…` |
| **`v0.30.0`（本分支）** | **官方预编译（python 3.12.16），md5 `72675e2b…`** |

本分支那份来自 `astral-sh/python-build-standalone`：

```
asset   cpython-3.12.16+20261026-aarch64-unknown-linux-gnu-install_only.tar.gz
sha256  a945427411dc6851a1393533b74a4a1ac434321ae397f2ec69897a9b4fde032b
url     https://github.com/astral-sh/python-build-standalone/releases/download/20261026/
        cpython-3.12.16+20261026-aarch64-unknown-linux-gnu-install_only.tar.gz
```

它是**货真价实的 PGO+LTO**（`--enable-optimizations --with-lto=full PROFILE_TASK=-m test --pgo`），
版本与镜像 `3.12.16` 精确一致。解包后取里面的 `libpython3.12.so.1.0` 使用
（`sha256` 不符别用）。`main` 分支那份由本机脚本现编，耗时约 1322 s。

- `rc2` 基座的 3.12.13 产物：实测每步 decode 时延 **−3.2~5.1%**（均值 −4.27%，6 档全降）。
- 本分支的 3.12.16 产物：**增益尚未实测**，需做 PGO on/off 同协议对照。
- ⚠ **落点必须探测，不能写死版本号**：路径里的 python 次版本与镜像不符时，
  docker 会自动建一个空目录、挂载"成功"、解释器按 DT_RPATH 加载镜像自带的库
  ⇒ PGO 完全失效，而 `docker inspect` / 启动日志 / `sys.version` 全都看不出异常。
  `PGO_DEST=` 可显式覆盖。

> PGO 收益只看**每步时延**。吞吐会被 `A_spec` 抽签噪声盖住，跨进程比吞吐会得出相反结论。

---

## 6. 可选：KV 外部池（`KV_OFFLOAD=1`）

把 KV 缓存卸载到主机 DRAM 池，跨请求复用。脚本会自动起 MetaService 容器、生成配置、
加 `--kv-transfer-config`。实现在同目录 `kv_offload.sh`（工作区根目录那条起服入口
也 source 它，所以两条入口的池行为完全一致）。

```bash
KV_OFFLOAD=1 bash start-server.sh
KV_OFFLOAD=1 KV_DRAM_GB=4 bash start-server.sh     # 改池大小
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `KV_DRAM_GB` | **2** | **每 rank**（不是整机）；8 卡 ⇒ 16 GB |
| `KV_PROTOCOL` | `host_shm` | 别改，见下 |
| `KV_WORLD_SIZE` | 256 | 可登记 rank 上限，改了要重启 meta |
| `KV_META_PORT` / `KV_STORE_PORT` / `KV_METRICS_PORT` | 5100 / 6100 / 8100 | metrics 不能是 8000，会和 `--port` 撞 |
| `KV_ALLOW_UNFIXED_TREE` | 0 | 见下"源码树要求"，默认不许绕过 |

MetaService 容器名 `<NAME>-mmc-meta`，不占卡。已存在时校验端口后复用（不重启，重启会丢池）。

**两个必须遵守的约束**：

1. `KV_PROTOCOL` 必须 `host_shm`。`host_rdma` 的 extend 库在镜像是 0 字节；
   `device_rdma` 单 rank 能过、**8 rank 必 segfault**。
2. 不要给容器加 `-e LD_LIBRARY_PATH`。它会覆盖镜像的 CANN 路径 ⇒ `libhccl.so` 找不到。

**源码树要求**：挂载树的 `pool_scheduler.py` 必须含 MRV2 抢占恢复后清
`_preempted_req_ids` 的那行代码（当前在 `pool_scheduler.py:825`，由上游
`vllm-ascend v0.30.0` 自带）。缺这行时，抢占后的池 key 会关联到**错误的 block**
—— 命中池就读到别人的 KV，输出错但日志不报错。

所以 `KV_OFFLOAD=1` 起服前会自动校验挂载树，缺这行直接 `exit 8`：

```
FATAL: KV_OFFLOAD=1 但源码树不含 KV 池的 MRV2 抢占恢复修复
```

判据是"那行代码在不在"，不关心由谁提供：`v0.30.0` 基座的树由上游代码满足。
真要不带修复跑：

```bash
KV_ALLOW_UNFIXED_TREE=1 KV_OFFLOAD=1 bash start-server.sh   # 会降级为警告
```

**确认真的生效**（`WAIT=1` 会自检；池是静默失效型，起得来不代表接上了）：

```bash
grep -c 'Creating v1 connector with name: AscendStoreConnector' <log>   # 期望 8
grep -c 'RegisterBaseInfo.*register capacity' <log>                     # 期望 8
curl -s http://127.0.0.1:8100/metrics | grep '^memcache_total_capacity_bytes'   # = KV_DRAM_GB × 8
```

命中效果看日志 `kvpool hit tokens` 和 metrics 里 `batch_get_*` / `evict_operations_total`。

> 池容量参考：全量 KV 约 `29963 block × 1 MiB = 29.26 GiB/机`，默认 16 GB 约占 55%。
> 该尺寸下跑一轮 8K 压测无驱逐。驱逐只丢块，不影响正确性。

> ⚠ 判正确性**必须用首 token**：同 prompt 连发多次，整段文本会有同义抖动，不能整段比对。

---

## 7. 6 个开关必须都传（外加 AIV）

下面 6 个 env **默认值全是 0**，不传时服务照常起、日志不报错，但收益全丢：

```
VLLM_ASCEND_V41_TRITON_SLOT_MAPPING     单 launch Triton 槽位映射
VLLM_ASCEND_ROPE_INDEX_SELECT           cos/sin 走 index_select
VLLM_ASCEND_V41_MOE_MASK_RANGE          连续 expert 区间比较
VLLM_ASCEND_V41_O_PROJ_2D               wo_a 退化 batch matmul → 2D matmul
VLLM_ASCEND_V41_QLI_NO_CANDIDATE        每层走 mode 3，关掉两级候选机制
VLLM_ASCEND_V41_ENGRAM_WKV_TP           engram wkv 按输出维分片（每卡省 550 MB 换 KV）
```

**全部在 import 期读取且在捕获图里 ⇒ 只能重启切换，热改 env 无效。**

脚本默认还会多传一个 **`HCCL_OP_EXPANSION_MODE=AIV`**（不是 `VLLM_ASCEND_*` gate，不走
`envs.py`，也不受 gate 定义校验管）。它把集合通信的展开/执行从 HOST 挪到 Vector Core，
实测通信 **−42.3%**、每步 **−10.8%**，零代码。两个前提：**单机对称 8 卡**、**非确定性**
（开了 `HCCL_DETERMINISTIC=true` 会被压掉）。回退就是把它从 `PASSTHRU_ENVS_STR` 里删掉。

> 另有 `MULTISTREAM_SHARED_EXPERT`（§4）也是同类"静默失效"开关，但它走
> `--additional-config`，不是 env，因此不在此表。

脚本有两道防护：起服前扫源码树的 `envs.py` 确认每个 gate 都有定义（缺了 `exit 3`）；
`WAIT=1` 时硬校验真的走了 V2。

---

## 8. 部署后验证

```bash
# 1) 服务活着
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/health   # 期望 200

# 2) 参数与 env 真的生效了
docker inspect dsv41 --format '{{json .Config.Cmd}}' | tr ',' '\n' | grep -E 'speculative|enable-prompt'
docker inspect dsv41 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'VLLM_ASCEND|HCCL_OP_EXPANSION_MODE'

# 2b) AIV 真的生效了（唯一可信判据，不用重启就能拿）
grep -rl 'aivMode\[1\]' \
  runtime/$(hostname -s)/dsv41/logs/ascend_plog/log/run/plog/ | wc -l   # 期望 8
#    完整自证行形如：
#    [HCCL_ENV] environmental variable HCCL_OP_EXPANSION_MODE is [AIV],
#               aicpuUnfold[0], aivMode[1], enableFfts[1], aicpuCacheEnable[0]
#    8 个 rank 各一行（主进程/EngineCore 那两个 plog 里没有，正常）。
#    全是 aivMode[0] 说明 env 没设上或被 HCCL_DETERMINISTIC 压掉。

# 3) 确认走的是 V2（V1 的判据在 V2 下恒为 0，不能拿来用）
docker logs dsv41 2>&1 | grep -c 'npu model runner v2 is in developing'   # 期望 8
docker logs dsv41 2>&1 | grep -c '\[dspark-graph-capture\]'                # 期望 0
#    目标图：日志里 "Capturing CUDA graphs (FULL): 100%|...| 32/32"
#    "ACL graph replay is active for the V2 target model" 在**首个请求之后**才打印，
#    起服完成时看不到是正常的
```

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `FATAL: 缺构建产物 ...vllm_ascend_C...so` | `SRC_DIR` 是空 git 树，不是从镜像取的。删掉重跑 |
| `FATAL: 这些 gate 传了 =1，但 envs.py 里没有定义` | 补丁没打成功，或 `SRC_DIR` 指向未打补丁的树 |
| 起服了但明显慢，日志没报错 | 6 个开关少了几个，用 §8 第 2 步核对 |
| `ImportError: libascend_hal.so` | 在裸 `docker run` 里 import vllm；必须带全部 `--device` 和驱动挂载 |
| 起不来但没留有用日志 | 脚本保留 log/plog/npulog，`docker stop` 后仍可查 `$LOG_DIR/logs/` |
| KV 容量上报每次差 ±128 | 固定 HBM 下按 `block_size=128` 取整，不是参数变化 |
| `FATAL: 端口 8000 已被占用` | 换 `PORT=`；或本机有别的吃 8 卡容器，先 `docker stop` |
| `vllm serve --help` 找不到 `--enable-prompt-tokens-details` | 顶层只列 Config Groups，要 `--help=Frontend` |

logs / plog / npulog / 编译缓存落在 `$LOG_DIR`（默认 `./runtime/<hostname>/<NAME>/`），按主机隔离。

---

## 10. 兼容性提醒

- **不要用 `vllm_ascend/_version.py` 判断实际跑的代码**：它由构建期生成，本补丁没重新生成，
  报的仍是镜像构建期那版。要看真实状态用
  `git -C $SRC_DIR/vllm-ascend log --oneline -1`（本分支应为 `c09d3e20b`）。
- 补丁针对 `v0.30.0`（`c09d3e20b`）原样应用，`series.conf` 里 5 个补丁全部干净通过。
  镜像里 `/vllm-workspace` 的树被改过则 `git apply` 失败，脚本立刻报错，不会静默。
- 只验证过 8 卡 TP8/EP8 单机，改 `TP` 未测。
- Model Runner V2 在 vllm-ascend 侧标注 `[Experimental]`；官方 DeepSeek-V4 + dspark 的 e2e
  覆盖的是 A3，A2 无官方覆盖。
- V2 图捕获规模：目标模型 32 张、draft 24 张（V1 是 draft 256 条）。
- `SPEC_TOKENS` 只能是 `n_predict=5` 的整数倍，取 10 实测更差。
- 跨会话比 `A_spec` 会得出相反结论（每步成本可复现到 0.3%，`A_spec` 只到 8~16%）。

---

## 11. 换基座（升级到新镜像）怎么做

**开新分支，不要原地改。** 一个分支 = 一个基座。

```bash
cd vllm-ascend-deepseek-v41-a2        # 本仓库根目录
git checkout -b v0310 main          # 从 main 或上一个基座分支拉

# 1) 只改 base.conf：镜像名 / 镜像 Id / 上游 commit / CANN / python / triton / 树目录
$EDITOR base.conf

# 2) 逐补丁试打；上游已自带的补丁不要写进 series.conf
for p in $(grep -v '^#' series.conf | grep -v '^$'); do
  git -C /tmp/probe apply --check patches/$p && echo "OK $p" || echo "FAIL $p"
done

# 3) 建树（★ 不重启、不动正在跑的服务）
MODEL=<权重目录> PREPARE_ONLY=1 bash start-server.sh

# 4) 校验生成的命令（不碰卡）
DRY_RUN=1 bash start-server.sh
```

`PREPARE_ONLY=1` 是专门为此加的：只建树 + 校验 gate 定义 + 探测 PGO 落点，
**不建容器、不碰端口、不动卡**，所以可以在服务照常跑的时候准备好新基座，
两棵树并存随时 A/B。真正切过去才需要重启（届时 `FORCE=1`）。

换基座时逐项确认（漏掉任何一项都是静默失效）：

- [ ] `base.conf` 的 `BASE_IMAGE_ID` 与 `docker image inspect` 实测一致
- [ ] `BASE_TREE_DIR` 换了新名字（否则切分支会原地覆盖另一条基座的树）
- [ ] `BASE_PGO_SO_REL` 指向与镜像 python 版本**匹配**的 PGO 产物
- [ ] `series.conf` 里没有上游已自带的补丁（否则 `git apply` 会失败）
- [ ] 建成后源码树里 `.patches-applied` 的第 3 行 = 本次镜像 Id
- [ ] 本 README 的实测数据表标注清楚基座，新基座的表重测后补齐

## 12. 补丁清单

`series.conf` 里 **5 个补丁 + 1 个 vllm 侧补丁**，按职责分两层：

| 补丁 | 作用 |
|---|---|
| `0001-envs-gates.patch` | 在 `envs.py` 定义 6 个 gate。**没有它其余补丁都不生效**（env 未定义会被 vLLM 静默忽略） |
| `0002-dsa-v41.patch` | `attention/dsa_v41.py`。V2 主路径优化：`_sentinel`/`_build_consts` 记忆化、`where(out=)` 原地写回、Triton 槽位映射、QLI no-candidate、槽位链组内复用 |
| `0003-dsa-v1-o-proj-2d.patch` | `attention/dsa_v1.py` 的 `wo_a` 由退化 batch matmul 改 2D matmul。改动由 `V41_O_PROJ_2D` 门控、**默认关闭** |
| `0004-model-opts.patch` | 模型侧五项：`engram/layer.py`（wkv TP 分片）、`indexer.py`（QLI）、`model.py`（wkv 分片镜像）、`fused_moe/token_dispatcher.py`（mask-range）、`ops/rope_dsv4.py`（index_select） |
| `0005-packed-cache-slot-mapping-scalar.patch` | `ops/triton/packed_cache_slot_mapping.py`。A2 的 AIV 标量单元用软件模拟 64 位整数除法，而这个内核只搬几百字节却 89% 时间卡在标量管线。改为 int32 计算，2 的幂 page 用移位/掩码（非 2 的幂保留除法路径） |
| `vllm.patch` | vllm 侧 `tokenizers/deepseek_v41_encoding.py`：控制 token 逃逸，防伪造轮次边界 |
