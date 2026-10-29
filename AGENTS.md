# AGENTS.md — 交接与协作说明

面向接手本仓库的**人或 AI agent**。README 讲「怎么部署」，本文讲「这个仓库内部
长什么样、改东西时踩哪些坑」。**动手前先读 §1 和 §5。**

基线：`vllm-ascend v0.30.0rc2`（`8fde6d4e3`）、`vllm v0.30.0`（`ced6857af`）。
默认启用 Model Runner V2。

---

## 1. 三条硬约束（违反会静默出错）

1. **补丁只能基于镜像里的源码树**，不能 `git clone` 上游再打。运行必需的 `.so`
   不在 git 里（被 `.gitignore` 排除），必须与镜像配对。构建产物缺失直接在 import
   期 `ImportError`，`start-server.sh` 已前置硬校验。
2. **改任何 patch 或 `series.conf` 都要重新准备树**。脚本用**序列指纹**
   （`series.conf` + 各 patch 的 md5 + `vllm.patch`）+ `$SRC_DIR/.patches-applied`
   判断是否重建。历史事故：v2 把补丁从 1788 行裁到 796 行，`tree/` 还是 v1 的
   941 行改动却照用不误。**别改成「看标记文件是否存在」那种判据。**
3. **6 个 gate 少传不会报错**。默认全 0，不传时服务照常起、日志一行不吭、收益全丢。
   `start-server.sh` 有两道防护（起服前扫 `envs.py` 定义、`WAIT=1` 硬校验 V2 标记），
   改动这些逻辑时不要削弱。

---

## 2. 仓库结构

### 2.1 补丁仓库内部结构

| 路径 | 作用 |
|---|---|
| `README.md` | 部署指引 + 实测数据。**给用户看的**，别塞内部细节 |
| `AGENTS.md` | 本文件。交接/协作说明、内部结构、踩坑记录 |
| `start-server.sh` | 起服脚本，唯一的执行入口（488 行） |
| `series.conf` | 补丁序列声明，`start-server.sh` 按行读取并顺序 apply |
| `patches/0*.patch` | vllm-ascend 侧性能/功能优化，**默认启用** |
| `patches/1*.patch` | vllm-ascend 侧 KV offload 相关，只有挂 `--kv-transfer-config` 时需要 |
| `vllm.patch` | vllm 侧补丁（1 文件，prompt 编码两个缺陷） |
| `tests/` | 回归用例（见 §2.3） |
| `legacy/` | 旧的单个大补丁 `vllm-ascend.monolithic.patch`（796 行），**仅供回退**，正常流程不读 |
| `tree/` | ← 运行期产物，`.gitignore` 排除。从镜像里取出的源码树 + 打完补丁的结果 |
| `runtime/` | ← 运行期产物，`.gitignore` 排除。日志/plog/npulog/编译缓存，按 `<hostname>/<NAME>` 隔离 |

`tree/` 和 `runtime/` **不在版本控制里**，删掉会自动重建（`tree/` 约 0.6 GB，
重建约 1 min）。**不要提交它们。**

### 2.2 每个 patch 的职责

分成 `0xxx`（性能/功能）和 `1xxx`（KV offload）两层，**目的是升级基线时能只重做
受影响的那层**。

| Patch | 文件数 | 改动量 | 职责 | 默认 |
|---|---|---|---|---|
| `0001-envs-gates.patch` | 1 | +39/−0 | 在 `vllm_ascend/envs.py` 定义 6 个 gate。**没有它其余补丁都不生效**（env 未定义会被 vLLM 忽略） | 启用 |
| `0002-dsa-v41.patch` | 1 | +174/−51 | `attention/dsa_v41.py`。V2 主路径：`_sentinel`/`_build_consts` 记忆化、`where(out=)` 原地写回、Triton 槽位映射、QLI no-candidate、槽位链组内复用 | 启用 |
| `0003-dsa-v1-o-proj-2d.patch` | 1 | +37/−1 | `attention/dsa_v1.py`。`wo_a` 退化 batch matmul → 2D matmul | 启用（改动由 `V41_O_PROJ_2D` 门控，默认关） |
| `0004-model-opts.patch` | 5 | +172/−12 | 模型侧五项：`engram/layer.py`（wkv TP 分片）、`indexer.py`（QLI）、`model.py`（wkv 分片镜像）、`fused_moe/token_dispatcher.py`（mask-range）、`ops/rope_dsv4.py`（index_select） | 启用 |
| `0101-kvpool-mrv2-resume.patch` | 1 | +9/−0 | `pool_scheduler.py`。MRV2 抢占恢复时 `_preempted_req_ids` 未清理，会把错误 block 关联到池 key（**静默算错**）。上游 `6197fec5c` / #19693 的等价修复，2 行 | 启用 |
| `vllm.patch` | 1 | +64/−3 | `tokenizers/deepseek_v41_encoding.py`：控制 token 注入（防伪造轮次边界）、空 developer 消息 | 启用 |

> **`0003` 不能因为「V2 默认走 `dsa_v41`」就删掉。** `dsa_v1.py` 在 V2 下确实会被
> import（`worker/v2/attn_utils.py:64` 引用 `AscendDSAMetadataBuilder`）。它的改动由
> `VLLM_ASCEND_V41_O_PROJ_2D` 门控、默认关闭，所以升级基线时可以**延后**处理。

### 2.3 `tests/`

| 文件 | 作用 |
|---|---|
| `tests/ut/distributed/ascend_store/test_mrv2_preempt_marker.py` | `0101` 的定向回归。验证 MRV2 抢占标记被 `_process_new_request` 消费 |

**注意**：该用例 import 了 `tests.ut.distributed.ascend_store._mock_deps`，而
**这个文件目前不在仓库里**（也无 `__init__.py`）。要在仓库内直接跑通需要补齐
mock 依赖，或者按 `b27a32d` 提交里记录的方式，把它拷进 vllm-ascend 源码树的
`tests/` 下运行。

### 2.4 分层原则（升级基线时用）

`series.conf` 头部记录了上游 `v0.30.0rc2`(8fde6d4e3) → main 452 提交的实测 churn：

| 文件 | 上游 churn | 含义 |
|---|---|---|
| `dsa_v1.py` | 11 提交 +623/−159 | **冲突重灾区**，但改动默认关闭，可延后 |
| `dsa_v41.py` | 5 提交 +153/−4 | 需重点评估 |
| `model.py` | 7 提交 +46/−1 | 需评估 |
| `rope_dsv4.py` | 1 提交 | 轻 |
| `engram/layer.py` / `indexer.py` / `token_dispatcher.py` | 0 提交 | **零冲突** |

升级到新基线（例如 0.31.0rc1）时：先重做 `0001`（纯新增，几乎不会冲突），
再按上表自下而上处理。

---

## 3. `start-server.sh` 结构

488 行，关键段落（行号会随改动漂移，按注释标题搜）：

| 段 | 内容 |
|---|---|
| `必填项` | `IMAGE`、`MODEL` 必须由环境变量传入 |
| `可调项` | 24 个默认赋值（其中 ~11 个是用户常调的，见 README §4；其余是 KV 池细节） |
| `KV 外部池` | `KV_OFFLOAD=1` 才启用，`KV_DRAM_GB` 是**每 rank**不是整机 |
| `PASSTHRU_ENVS_STR` | 6 个 gate 的 `-e` 串 **+ `HCCL_OP_EXPANSION_MODE=AIV`**（共 7 条）。前 6 条受 gate 定义校验，AIV 那条不受（校验只抓 `VLLM_ASCEND_*=1`）。**改这里要同步改 §6 的说明** |
| `已有一个吃 8 卡的容器就别重复起` | 前置拦截 + `ALLOW_FOREIGN` 白名单 |
| `源码树` | 序列指纹判据 + 从镜像 tar + 按 series apply + 构建产物校验 + gate 定义校验 |
| `KV 外部池`（第二段） | MetaService 容器起/复用、生成配置、就绪闸门 |
| `WAIT=1` | 等 READY + 硬校验 V2 判据 + KV 判据 |

### 退出码

| 码 | 含义 |
|---|---|
| 1 | 前置条件不满足（缺 `MODEL/config.json`、缺命令、缺镜像、缺构建产物、apply 失败、`PGO_SO` 不存在、容器已在跑） |
| 2 | 端口占用 / 已有其他 8 卡容器 |
| 3 | gate 在 `envs.py` 里没定义（补丁没打成功）／`WAIT` 宽限期内容器未创建 |
| 4 | 日志里没有 V2 worker 标记（实际走的可能是 V1） |
| 6 | KV MetaService 相关（端口不一致 / metrics 60s 内不通 / 就绪失败） |
| 0 | `DRY_RUN=1` 正常结束 |

### 两点易错处

- **`set -euo pipefail` 下 `cat missing | head` 会静默退出**。已有先例（`HAVE_FP`
  取值），改脚本时凡是可能读不存在的文件都要 `|| true` 兜住。
- **`ALLOW_FOREIGN` 默认放行 `deepseek-v4`**。那是 2026-08 起的陈旧容器，只占设备
  节点、不做推理，**与本工作区无关，别去动它**。真要连它一起管就置空。

---

## 4. 6 个 gate 与 V1 的差异

```
VLLM_ASCEND_V41_TRITON_SLOT_MAPPING     单 launch Triton 槽位映射
VLLM_ASCEND_ROPE_INDEX_SELECT           cos/sin 走 index_select
VLLM_ASCEND_V41_MOE_MASK_RANGE          连续 expert 区间比较
VLLM_ASCEND_V41_O_PROJ_2D               wo_a 退化 batch matmul → 2D matmul
VLLM_ASCEND_V41_QLI_NO_CANDIDATE        每层走 mode 3，关掉两级候选机制
VLLM_ASCEND_V41_ENGRAM_WKV_TP           engram wkv 按输出维分片（每卡省 550 MB 换 KV）
```

**全部在 import 期读取且在捕获图里 ⇒ 只能重启切换，热改 env 无效。**

> 与这 6 个 gate **并列但不同类**的还有一条 `HCCL_OP_EXPANSION_MODE=AIV`（同样在
> `PASSTHRU_ENVS_STR` 里、同样默认开、同样重启才生效）：它不在 `envs.py` 里、不需要任何
> patch，改的是 HCCL 通信层。所以 `patches/` 与 `series.conf` 一行都没动，
> 但**它是默认档的一部分** —— 复现实测数据时必须带上，否则会慢约 10%。

V1 分支有 **12** 个，本文档对应的 V2 分支只留 **6** 个。另外 6 个
`VLLM_ASCEND_DSPARK_*` 只在 `spec_decode/dspark_proposer.py` 与
`llm_base_proposer.py` 里被读，而这两个文件属于 V1 的 proposer 路径。V2 走
`worker/v2/spec_decode/dspark/speculator.py`（继承上游 `DSparkSpeculator` +
`DFlashAclGraphManager`），根本不 import 它们。V1 里那 6 个开关换来的「draft 进
ACLGraph」，在 V2 由 `query_cudagraph_manager` 原生提供。**传了不报错也不生效，
所以本分支既不定义也不传。**

V1 的完整内容保存在 `v1` 分支（`3d429b6`），补丁为单文件 1788 行。

---

## 5. 已知坑与历史事故

| 现象 | 根因 | 处理 |
|---|---|---|
| 改了补丁但效果没变 | 序列指纹没变或 `tree/` 是旧的 | 删 `$SRC_DIR/.patches-applied` 强制重建；核对 fingerprint |
| 起服成功但明显慢、日志无报错 | 6 个 gate 少了几个 | 按 README §8 第 2 步核对容器 env |
| `FATAL: 缺构建产物 ...vllm_ascend_C...so` | `SRC_DIR` 是空 git 树，不是从镜像取的 | 删掉重跑 |
| `FATAL: 这些 gate 传了 =1，但 envs.py 里没有定义` | 补丁没打成功，或 `SRC_DIR` 指向未打补丁的树 | 查 `series.conf` 与 apply 日志 |
| `ReqMeta GVA metadata does not cover requested block range` | 缺 `0101`，或该 patch 未应用 | 确认 `0101-kvpool-mrv2-resume.patch` 在 series 里且 apply 成功 |
| KV 池起得来但没接上 | 池是**静默失效型** | `grep -c 'Creating v1 connector'` 期望 8；见 README §6 |
| 端口 8000 被占 | 本机有别的吃 8 卡容器（如 `dsv41-rc2`） | 换 `PORT=`；或先停掉；必要时调 `ALLOW_FOREIGN` |
| KV 容量每次差 ±128 | 固定 HBM 下按 `block_size=128` 取整 | 不是参数变化，正常 |
| `/start_profile` 返回 404 | 起服时没带 `--profiler-config` | 用 `PROFILER=1` 起服（根脚本透传给 `rc2/serve.sh`）|
| 剖面工具报 `no per-step marker kernel found` | 工具里写死了 **V1** 的 marker 名 | V2 是 `_compute_slot_mappings_kernel`，见 `rc2/bench/step_markers.py` |
| 剖面 span 只有 0.017 ms、hole≈0 | 把两个 V2 marker **混用**了 | V2 两个 marker 同一步内相距 0.017 ms，只能取**单一序列** |
| 剖面每步 285 ms 而不是 58 ms | prefill 也触发了同一个 marker | 用 `--decode-only` / `rank_balance.py` 的 decode 过滤 |
| `overlap.py` 报 `need at least three step barriers, found 0` | V2 trace 里没有 `Event::synchronize` | 该工具仅适用 V1；V2 用 `gap_attr.py` |

### 剖面口径（2026-10-28 定）

`PROFILER=1` 不改变编译/图（`ProfilerConfig.compute_hash()` 的 factors 为空），但
**profiled 窗口的每步时延是虚高的，不能拿去和未插桩的运行比**——只用来看构成和占比。

- 窗口要等**全部流首 token 到齐**再开（`rc2/bench/probe_prof.py`），否则窗口里混进 prefill
- trace 写 **宿主 `/tmp`**（容器已 `-v /tmp:/tmp`）：8 rank × ~700 MB，写 NFS 会卡
- `torch_npu.profiler.profiler.analyse()` 必须在**容器内**跑（要 torch_npu），单 rank 约 77 s
- c=16 实测：设备空档 2.5 ms/步、被 host 覆盖的仅 0.29 ms ⇒ **host 侧已无可削空间**，
  且通信与计算**零重叠**（10 ms/步全在关键路径上）

### 收益开关（2026-10-29 定）

| 开关 | 状态 | 生效自证 |
|---|---|---|
| `MULTISTREAM_SHARED_EXPERT=1`（默认） | ✅ **已开**，每步 −1.3~−4.6% | `shared_experts.py:180` 的 `Shared expert split computation validation passed` |
| **`HCCL_OP_EXPANSION_MODE=AIV`（默认）** | ✅ **已开**，通信 −42%、每步 −10.8% | plog `[HCCL_ENV] ... aivMode is [1]`（**8/8 rank**，默认 `HCCL_DEBUG_CONFIG=0x0` 就会打印） |
| `enable_flashcomm1=true` | ❌ **已否决**，+8~+22% | 不要开：会砍掉非 TP 倍数的 capture size（32/24→24/20），且延迟受限下拆通信是净亏 |

- 多流是**上游镜像自带的配置项**，不涉及本仓库任何 patch；AIV 是 **HCCL 侧 env，同样零代码**，
  但它不在 `envs.py` 里，也不走 `--additional-config`，而是和 6 个 gate 一起放在 `PASSTHRU_ENVS_STR`。
- ⚠ **AIV 与 `HCCL_DETERMINISTIC=true` 冲突**（确定性优先会压掉 AIV），当前非确定性，别打开它。
- ⚠ **AIV 会多吃一点 device workspace**：同一配置可用 KV 15.61 → 15.55 GiB、容量约 3.64M → 3.63M
  token（单次观测，未做隔离变量；仍 3.46x 并发，不影响可用性）。
- 加其它 `additional_config` 用 `ADDL_EXTRA='"k":v'`（`serve.sh` 会拼进 `ADDL_JSON`），
  先 `DRY_RUN=1` 验。`ADDL_EXTRA` 与 `MULTISTREAM_SHARED_EXPERT` 会自动合并。
- ⚠ `SRC_ROOT` 默认 `patch`（挂 `tree/`）；要和挂 `src/` 的基线对照必须显式 `SRC_ROOT=src`。

### 历史事故（改脚本/补丁前值得知道）

- **补丁静默用旧树**（`b27a32d`）：旧判据只看 `.patches-applied` 是否存在，
  v2 裁剪补丁后 `tree/` 仍是 v1 内容却照用。现改为序列指纹。
- **`FORCE=1` 停不掉别的名字的 8 卡容器**（`d0904d8`）：两个容器同时抢 8 卡和
  8000 端口，起服以各种奇怪方式失败（HBM 不够 / HCCL 超时）。现加前置拦截。
- **MRV2 抢占恢复算错**（`b27a32d`）：`_preempted_req_ids` 未清理，把错误 block
  关联到池 key，**静默算错**而非报错。`0101` 修复。

---

## 6. 改东西时的检查清单

改 **patch** 或 **series.conf**：
- [ ] 序列指纹会变，确认重建后 `tree/` 里 `git apply` 全部干净
- [ ] 若新增 gate，同步改 `PASSTHRU_ENVS_STR`（§3）和 `envs.py`（`0001`）
- [ ] 若新增 gate，同步改 README §7 与本文档 §4

改 **start-server.sh**：
- [ ] 保持 `set -euo pipefail` 安全（读可能不存在的文件要 `|| true`）
- [ ] 退出码语义别改（§3 那张表是本文档独有的，README 未列）
- [ ] `WAIT=1` 的 V2 判据别退回 V1 判据 —— V1 的 `runtime_mode=FULL` /
      `[dspark-graph-capture]` 在 V2 下**恒为 0**，照抄会把成功判成失败
- [ ] 动 `PASSTHRU_ENVS_STR` 时**保留 `HCCL_OP_EXPANSION_MODE=AIV`**。它是纯 `-e` 串、
      不经过 gate 校验，删掉不会报任何错，只会让每步慢 10%（实测过）

改 **README**：
- [ ] 用户向内容放 README，内部结构/踩坑放本文档
- [ ] 实测数据表格别删（`7d2eaf4` 曾误删又补回）
- [ ] 性能结论只用**每步时延**，不要用吞吐（受 `A_spec` 抽签影响 ±15%）

---

## 7. 分支

| 分支 | 内容 |
|---|---|
| `main` | 当前（V2）。补丁拆分为 `series.conf` + `patches/`，含 KV offload |
| `v1` | V1 旧版（`3d429b6`）。单文件 `vllm-ascend.patch` 1788 行，12 个 gate，`VLLM_USE_V2_MODEL_RUNNER=0` |
| `v2` | 与 `main` 同 commit 的历史别名，保留兼容 |

---

## 8. 没有验证过的部分

诚实记录边界，别把以下当成已验证：

- **只验证过 8 卡 TP8/EP8 单机**。改 `TP` 未测。
- Model Runner V2 在 vllm-ascend 侧标注 `[Experimental]`，官方 DeepSeek-V4 + dspark
  的 e2e 覆盖的是 **A3**，**A2 无官方覆盖**。
- `SPEC_TOKENS` 只能是 `n_predict=5` 的整数倍，取 10 实测更差。
- **AIV 的收益只在「A2 + 单机 8 卡 + 对称推理」上测过**（c=1/4/5/8/9/13/16 全档都测了）。
  A3、多机、PD 分离都没测；官方文档还说 A2 的 AIV「不支持多通信域并行」——本机虽然
  实测没触发（3 个 8-rank 域下全局开也正常），但别的拓扑别照抄这个结论。
- **AIV 会让 KV 容量少约 12k token**（3.64M → 3.63M，可用 KV 15.61 → 15.55 GiB）。
  只观测到一次，没做隔离变量；仍 3.46x 并发，不影响可用性。
- 不要用 `vllm_ascend/_version.py` 判断实际跑的代码 —— 它由构建期生成，本补丁没
  重新生成，仍报 `0.30.0rc2` / `g8fde6d4e3`。要看真实状态用
  `git -C $SRC_DIR/vllm-ascend log --oneline -1`。
- `tests/` 的用例依赖的 `_mock_deps` 缺失，仓库内不可直接跑（§2.3）。
