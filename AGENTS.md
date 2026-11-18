# AGENTS.md — 交接与协作说明

面向接手本仓库的**人或 AI agent**。README 讲「怎么部署」，本文讲「这个仓库内部
长什么样、改东西时踩哪些坑」。**动手前先读 §1 和 §5。**

基线：`vllm-ascend v0.32.0rc1`（`24487f4f4`）、`vllm v0.32.0`（`5df63a7203`）、
CANN `9.2.1`、python `3.12.16`。默认启用 Model Runner V2。

> **分支 = 基座**。`main`/`v1`/`v2` 挂 `v0.30.0rc2`（`8fde6d4e3`，树目录 `tree/`，
> PGO 产物在仓库外的 `rc2/pgo/optim/pgo/`）；`v0300` 挂 `v0.30.0`（`c09d3e20b`，
> 树目录 `tree-v0300/`）；本分支 `v0320` 挂 `v0.32.0rc1`（`24487f4f4`，树目录 `tree-v0320/`）。
> `v0300` 与本分支的 PGO 都在仓库外的 `rc2/pgo/optim/pgo-3.12.16/`（两个基座 python 同为 3.12.16）。
> 三个基座的产物并存、互不覆盖。树目录名与 PGO 相对路径都由 `base.conf` 给出。
> 基座身份唯一来源 = `base.conf`；三个起服入口都从它取值，**不要在别处写死镜像名
> 或 python 版本号**（写死 python 版本会让 PGO 静默失效，见 §5.1）。

---

## 1. 四条硬约束（违反会静默出错）

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
4. **`README.md` 只描述现状，不写历史**。
   README 是"这台机器此刻怎么部署、现在是什么状态"的说明书，读者照着它操作。
   所以**禁止**出现任何"过去怎样"的表述：不要写"XX 已退役"、"以前是 YY"、
   "历史上曾 ZZ"、"从 A 改成了 B"、"rc2 时代是 …"。这类内容属于**变更过程**，
   归 `git log` 或 `rc2/docs/` 的结论文档，不归 README。

   由此推出的两条具体规则：
   - **不在用的东西直接不写**，不写"已废弃/已移除"的条目。例如某补丁不再应用，
     就把它的行从 README 的补丁清单里删掉 —— 而不是留一行加删除线再加一段说明。
     （补丁文件本身仍在 `patches/` 里、由 `series.conf` 决定用不用，那是代码事实，
     不需要 README 复述它的"退役史"。）
   - **唯一的例外是"实测数据"表**。表格里可以同时存在不同基座/不同配置的数字行，
     因为那些数字就是用来做**当前与过去对比**的。但每条数字必须标注清楚它属于哪个
     基座/配置，不能让人误以为是当前档位的读数。
   - **README 里只写本仓库里真实存在的路径**。`tree*/` 与 PGO `.so` 都是
     `.gitignore` 排除的运行期/编译产物，**不在仓库里**：树直接不提；PGO 只说
     "取自哪个上游 release + asset + sha256"，不写仓库内的相对路径。
     （`base.conf` 里的 `BASE_TREE_DIR` / `BASE_PGO_SO_REL` 是**配置值**，可以有；
      README 不该把它们的值当成本仓库的目录结构来展示。）

   ⚠ 这条规则的代价：README 不再告诉你"为什么现在是这样"。补那个信息的是
   `AGENTS.md §5 已知坑与历史事故` 和 `rc2/docs/` —— 想知道前因后果去那里，
   想知道现在怎么做看 README。

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
| `tree/`、`tree-v0300/`、`tree-v0320/` | ← 运行期产物，`.gitignore` 排除。从镜像里取出的源码树 + 打完补丁的结果，目录名由 `base.conf` 的 `BASE_TREE_DIR` 按基座分家 |
| `runtime/` | ← 运行期产物，`.gitignore` 排除。日志/plog/npulog/编译缓存，按 `<hostname>/<NAME>` 隔离 |

`tree*/` 和 `runtime/` **不在版本控制里**，删掉会自动重建（一棵树约 0.6 GB，
重建约 1 min）。**不要提交它们。**

> **PGO 产物不在本仓库里**，放在仓库外的 `rc2/pgo/optim/`（编译产物，按 python
> 版本分家，见 §7.1）。`base.conf` 的 `BASE_PGO_SO_REL` 以相对路径指过去。
> 产物来源与验证记录见 `rc2/docs/PGO_31216.md`，基座对照数据见 `rc2/docs/V0300_BASELINE.md`。

### 2.2 每个 patch 的职责

分成 `0xxx`（性能/功能）和 `1xxx`（KV offload）两层，**目的是升级基线时能只重做
受影响的那层**。

| Patch | 文件数 | 改动量 | 职责 | 默认 |
|---|---|---|---|---|
| `0001-envs-gates.patch` | 1 | +39/−0 | 在 `vllm_ascend/envs.py` 定义 6 个 gate。**没有它其余补丁都不生效**（env 未定义会被 vLLM 忽略） | 启用 |
| `0002-dsa-v41.patch` | 1 | +174/−51 | `attention/dsa_v41.py`。V2 主路径：`_sentinel`/`_build_consts` 记忆化、`where(out=)` 原地写回、Triton 槽位映射、QLI no-candidate、槽位链组内复用 | 启用 |
| `0003-dsa-v1-o-proj-2d.patch` | 1 | +37/−1 | `attention/dsa_v1.py`。`wo_a` 退化 batch matmul → 2D matmul | 启用（改动由 `V41_O_PROJ_2D` 门控，默认关） |
| `0004-model-opts.patch` | 5 | +172/−12 | 模型侧五项：`engram/layer.py`（wkv TP 分片）、`indexer.py`（QLI）、`model.py`（wkv 分片镜像）、`fused_moe/token_dispatcher.py`（mask-range）、`ops/rope_dsv4.py`（index_select） | 启用 |
| `0005-packed-cache-slot-mapping-scalar.patch` | 1 | +46/−7 | `ops/triton/packed_cache_slot_mapping.py`。A2 的 AIV 标量单元用软件模拟 64 位整数除法，而这个内核只搬几百字节却 89% 时间在标量管线。改为 int32 计算 + 2 的幂 page 用移位/掩码（page 96/100 保留除法路径，由 constexpr `POW2_PAGE` 门控） | 启用 |
| `0101-kvpool-mrv2-resume.patch` | 1 | +9/−0 | `pool_scheduler.py`。MRV2 抢占恢复时 `_preempted_req_ids` 未清理，会把错误 block 关联到池 key（**静默算错**）。上游 `6197fec5c` / #19693 的等价修复，2 行 | 启用 |
| `vllm.patch` | 1 | +64/−3 | `tokenizers/deepseek_v41_encoding.py`：控制 token 注入（防伪造轮次边界）、空 developer 消息 | 启用 |

> **`0003` 不能因为「V2 默认走 `dsa_v41`」就删掉。** `dsa_v1.py` 在 V2 下确实会被
> import（`worker/v2/attn_utils.py:64` 引用 `AscendDSAMetadataBuilder`）。它的改动由
> `VLLM_ASCEND_V41_O_PROJ_2D` 门控、默认关闭，所以升级基线时可以**延后**处理。

### 2.3 `tests/`

| 文件 | 作用 |
|---|---|
| `tests/ut/distributed/ascend_store/test_mrv2_preempt_marker.py` | `0101` 的定向回归。验证 MRV2 抢占标记被 `_process_new_request` 消费 |
| `tests/ut/ops/triton/test_packed_cache_slot_mapping.py` | `0005` 的定向回归，33 项。7 组 page/ratio/skip/dtype 组合 × 4 种 token/req 形状对比 eager 参考；含 `test_padded_tail_is_masked_for_ratio1`（把内核既有语义钉死）与坏参数拒绝 |

**注意**：该用例 import 了 `tests.ut.distributed.ascend_store._mock_deps`，而
**这个文件目前不在仓库里**（也无 `__init__.py`）。要在仓库内直接跑通需要补齐
mock 依赖，或者按 `b27a32d` 提交里记录的方式，把它拷进 vllm-ascend 源码树的
`tests/` 下运行。

### 2.4 分层原则（升级基线时用）

`series.conf` 头部记录了本轮 **`v0.30.0rc2`(8fde6d4e3) → `v0.30.0`(c09d3e20b)**
共 **320 提交**的实测 churn（2026-11-02 逐文件比对两个镜像得出，非估算）：

| 文件 | 上游 churn | 含义 |
|---|---|---|
| `envs.py` | 0 | 逐字节相同；6 个 gate 仍不存在 ⇒ `0001` 必须留 |
| `dsa_v41.py` | 0 | 逐字节相同 |
| `model.py` | 0 | 逐字节相同 |
| `indexer.py` / `engram/layer.py` / `rope_dsv4.py` / `token_dispatcher.py` | 0 | **零冲突** |
| `attention_v1.py` / `packed_cache_slot_mapping.py` | 0 | 逐字节相同（`0005` 原样可打） |
| `dsa_v1.py` | +11/−14 | 全是上游自用重构（`full_graph_mode` 显式入参、o_proj 缓冲按 `get_potential_max_tokens()`），与 `0003` 无关 |
| `pool_scheduler.py` | +44/−74 | **吸收了我们原来的 `0101`**（`1b06b6cc6`） |

结论：**12 个补丁目标文件里 10 个逐字节未变**，只有 `dsa_v1.py` 有无关重构、
`pool_scheduler.py` 吸收了 `0101`。所以本轮升级成本极低（5 个 0xxx 全部干净应用）。

#### 2.4.1 v0.30.0(9.2.0) → v0.32.0rc1：700 提交

2026-11-18 逐文件比对两个镜像得出（非估算）。**工具链没动**（CANN 9.2.1、python 3.12.16、
triton-ascend 3.6.0.post2 全同），但上游代码动得多：

| 文件 | 上游 churn | 处理 |
|---|---|---|
| `envs.py` | +0/−13 | **删掉了 `_strict_binary_env()` 与两个旧条目（正是 `0001` 的上下文锚点）** ⇒ `0001` 改为自带 `_v41_binary_env()` |
| `attention/dsa_v41.py` | +146/−6 | 新增 SWA bounded-replay 断言，位置与 `0002` 改写的那段配置查询重叠 ⇒ 人工合并（缓存值 + 保留断言） |
| `attention/dsa_v1.py` | +1170/−469 | 新增整套 OTP（o-proj TP）；`0003` 只是插入点撞车 ⇒ 两个方法都留 |
| `models/deepseek_v41/model.py` | +49/−21 | `enable_dsa_cp` 下线、加 `@support_torch_compile`、`OProjBufferPool`、`decoder_replay_layers`；自动合并 |
| `models/deepseek_v41/indexer.py` | +12/−2 | 新增 JIT warmup 注册；自动合并 |
| `ops/fused_moe/token_dispatcher.py` | +35/−8 | 新增 MX quant `CEIL` 路径；自动合并 |
| `packed_cache_slot_mapping.py` / `engram/layer.py` / `rope_dsv4.py` | 0 | 逐字节相同，`0005` / `0004` 原样可打 |
| `pool_scheduler.py` | +120/−372 | `0101` 的修复仍在（新树 `:555` 且注释写明 MRV2 resume）⇒ 继续退役，`kv_offload.sh` 闸门不用改 |
| `vllm/tokenizers/deepseek_v41_encoding.py` | +2/−2 | `REASONING_EFFORT_MAPPINGS` 里 `low` 25→50、`high` 50→75；`vllm.patch` 照常应用 |

两处**新的**、跨基座通用的事故形态（已修，见 §5「历史事故」）：

1. 上游把我们**踩着的辅助函数**删了 —— 补丁的 lambda 懒求值 ⇒ 不 import 失败，只在
   第一次读 gate 时 `NameError`。教训：**自己依赖的辅助函数要自己带**。
2. 上游删了运行期产物 `vllm/vllm/_rust_tool_parser.abi3.so`（0.32 起只有 `vllm-rs`），
   而 `start-server.sh` 把它写成必需项 ⇒ 建树阶段误报"缺构建产物"。教训：
   **产物校验要跨基座成立**（现在只硬校验两者都在的 `vllm-rs`）。

⚠ **判断"要重做多少"不要用 `git apply --check`**：本轮它对 6 个补丁报 5 个失败，
但三路复演（把我们的改动当一次提交 rebase 到新基座）显示**只有 3 个文件真冲突**，
其余全是上下文漂移假象。流程改为：

```bash
# 三路复演：base30 → ours → base32，然后把 ours rebase 到 base32
git init -q -b main . && cp -a <旧基座目标文件> . && git commit -m base30 && git tag base30
<应用现有补丁> && git commit -m ours && git tag ours
git checkout base30 && cp -a <新基座同名文件> . && git commit -m base32 && git tag base32
git checkout -b replay ours && git rebase base32     # ← 冲突数才是真实工量
```

之后照 `README.md` §11 的流程走：改 `base.conf` → 上述复演重做补丁 →
`PREPARE_ONLY=1` 建树 → `DRY_RUN=1` 校验命令 → 起服。

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
  后记（v0300）：上游 `1b06b6cc6` 已含同样 2 行，`0101` 退役。闸门仍在
  （`kv_offload.sh` 判"那行代码在不在"，不关心谁写的），所以将来上游改掉它会照样拦住。
- **PGO 挂载点写死 python 版本 ⇒ 静默失效**（v0300 升级时发现，**最阴的一类**）：
  `serve.sh` 与 `start-server.sh` 原本都写死
  `/usr/local/python3.12.13/lib/libpython3.12.so.1.0`。基座换到 `v0.30.0` 后
  镜像 python 变成 **3.12.16**，于是：
  docker 自动创建那个不存在的父目录 → 挂载**"成功"** → 解释器按 DT_RPATH 加载
  镜像自带的库 → **PGO 完全没生效**。
  而 `docker inspect`、启动日志、`sys.version` 全都看不出异常（不逐条 `ldd`
  根本发现不了），白丢 −3.2~5.1%。
  现改为 `base.sh` 的 `base_pgo_dest` **起一次性容器探测镜像内 LIBDIR**（0.7 s），
  探测失败会显式告警而不是静默挂空目录。`PGO_DEST=` 仍可显式覆盖。
  > 教训：凡"宿主机路径 → 容器内路径"的挂载，只要容器内路径带版本号，
  > 就必须探测、不能写死。同类还有 `--entrypoint` 的绝对路径、`LD_PRELOAD` 的库名。
- **补丁依赖上游辅助函数 ⇒ 上游删掉后 gate 静默变 0**（v0320 升级时发现）：
  `0001` 的 6 个 gate 都写 `lambda: _strict_binary_env(...)`，而 `_strict_binary_env`
  是**上游 `envs.py` 里的函数**。v0.32.0rc1 把它连同两个旧条目一起删了。
  由于 lambda 是懒求值，`import` 不报错、服务照常起、`docker exec env | grep` 也看得到
  变量 —— 只有在**第一次读某个 gate** 时抛 `NameError`，或者（更糟）被调用方吞掉后
  走默认关路径。
  现改为补丁自带 `_v41_binary_env()`（自己的代码，上游怎么改都不影响）。
  > 教训：补丁里**不要借上游的内部辅助函数**，宁可复制一份 8 行的。判据：这个符号上游
  > 没在公开 API 里承诺过，就不要依赖它。
- **运行期产物清单写死 ⇒ 新基座误报"缺构建产物"**（v0320 升级时发现）：
  `start-server.sh` 曾硬校验 `vllm/vllm/_rust_tool_parser.abi3.so`。v0.32.0rc1 起上游
  **删掉了这个 Python 扩展**（全树无任何 import 引用，`tools/build_rust.py` 里只剩一行
  注释），只留 `vllm-rs` 可执行文件 ⇒ 建树成功后立刻 `FATAL: 缺构建产物`。
  现改为硬校验三个基座都有的那几件（两个 vllm-ascend `.so` + `vllm-rs`）。
  > 教训：这类"从镜像里取出的产物"清单会随基座变。新增校验时先问一句
  > "这个东西在**所有**在用基座里都存在吗"，否则它就是把换基座变成一次故障。

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

| 分支 | 基座 | 内容 |
|---|---|---|
| `v0320` | **`v0.32.0rc1` (24487f4f4)** | ★ **当前**。CANN 9.2.1 / python 3.12.16 / triton 3.6.0.post2。树目录 `tree-v0320/`（`BASE_TREE_DIR`），PGO 用仓库外 `rc2/pgo/optim/pgo-3.12.16/`。5 补丁 + `vllm.patch`（已按新基座重做）。**尚未起服实测** |
| `v0300` | `v0.30.0` (c09d3e20b) | 上一轮基座，保留为可复现的对照臂。树目录 `tree-v0300/`，PGO 同上。5 补丁 + `vllm.patch` |
| `main` | `v0.30.0rc2` (8fde6d4e3) | rc2 基线（保留为可复现的对照臂）。补丁拆分为 `series.conf` + `patches/`，6 补丁含 `0101`；树目录 `tree/`，PGO 用仓库外 `rc2/pgo/optim/pgo/` |
| `v1` | `v0.30.0rc2` | V1 旧版（`3d429b6`）。单文件 `vllm-ascend.patch` 1788 行，12 个 gate，`VLLM_USE_V2_MODEL_RUNNER=0` |
| `v2` | `v0.30.0rc2` | 与 `main` 同 commit 的历史别名，保留兼容 |

### 7.1 为什么按基座开分支，而不是原地改

一个分支 = 一个基座。三个原因，每个都对应一次真实会踩的坑：

1. **实测数据依附基座**。README 那两张表是在 rc2 上量的；原地换基座后旧数字留在
   同一张表里，读的人无法分辨哪些还成立 —— 那是"看起来有效、其实换了底"的假数据。
2. **`tree/` 是 gitignored 单份产物**。两分支共用 `tree/` 时，切分支就会原地重建、
   把另一条基座的树覆盖掉，旧基座从此不可复现。所以 `base.conf` 的 `BASE_TREE_DIR`
   按分支分家（`tree/` / `tree-v0300/` / `tree-v0320/`）。
3. **PGO 产物按 python 版本分家**。3.12.13 与 3.12.16 的产物互不兼容，同一目录会互相
   覆盖。两份并存于**仓库外**的 `rc2/pgo/optim/`（`pgo/` 与 `pgo-3.12.16/`），
   `base.conf` 的 `BASE_PGO_SO_REL` 用相对路径指过去。

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
- **`v0320`（`v0.32.0rc1`）2026-11-18 已起服，自证逐条通过**：READY **685 s**、
  目标图 **32/32**、draft **24/24**、V1-only 标记 **0**、8 条 V2 worker 标记、
  6 个 gate 全传进容器、`aivMode[1]` **8/8 rank**、jemalloc 无 preload 告警、
  PGO 自证行 1 条、KV 池 `rank 登记 8/8`、KV 容量 **3,489,673 tokens**（`max_len=524288`）、
  冒烟三项全对。上游新增的 `@support_torch_compile` / `decoder_replay` **没有改变图捕获数量**。
  **仍未验证**：6 个 gate 与 PGO 在这个基座上的**实际收益**、两套协议的**每步时延**
  —— 要谈数字先按 README §11 测。
- **`TRITON_ASCEND_ARCH` 在 0.32 的 triton 里已弃用**（每次起服 8 行
  `FutureWarning: ... is deprecated ... host environment overrides are ignored`）。
  `serve.sh` 仍在传它，但**已被忽略**（arch 由设备决定）。无害，但别再指望它生效。
- 不要用 `vllm_ascend/_version.py` 判断实际跑的代码 —— 它由构建期生成，本补丁没
  重新生成，仍报镜像构建期那版。要看真实状态用
  `git -C $SRC_DIR/vllm-ascend log --oneline -1`。
- `tests/` 的用例依赖的 `_mock_deps` 缺失，仓库内不可直接跑（§2.3）。
