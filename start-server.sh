#!/usr/bin/env bash
# =============================================================================
# start-server.sh —— 起 DeepSeek-V4.1-Flash（vllm-ascend v0.30.0rc2 + v25 补丁）
#
# 单机 8 卡（Ascend 910B3 / A2）已验证配置。参数取自一次真实在跑的服务
# （docker inspect 逐项核对），不是从文档抄的。
#
#   IMAGE=... MODEL=... bash start-server.sh          # 首次会自动准备源码树
#   IMAGE=... MODEL=... DRY_RUN=1 bash start-server.sh # 只打印 docker 命令
#   IMAGE=... MODEL=... WAIT=1 bash start-server.sh    # 前台等到 READY 并自检
#   IMAGE=... MODEL=... FORCE=1 bash start-server.sh   # 先停旧容器
#   BAT=8192 bash start-server.sh                      # 任何值都可用同名 env 覆盖
#   KV_OFFLOAD=1 bash start-server.sh                  # 挂 KV 外部池（默认 2GB/卡）
#
# 必填：IMAGE（v0.30.0rc2 镜像）、MODEL（权重目录）
# 停服务：docker stop "$NAME"    健康：curl -s -o /dev/null -w '%{http_code}\n' \
#                                      http://127.0.0.1:8000/health
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

# ---- 必填项 --------------------------------------------------------------
: "${IMAGE:?请用 IMAGE=<v0.30.0rc2 镜像> 指定镜像，例如 IMAGE=quay.io/ascend/vllm-ascend:v0.30.0rc2}"
: "${MODEL:?请用 MODEL=<权重目录> 指定模型，目录下必须有 config.json}"
[[ -f "$MODEL/config.json" ]] || { echo "FATAL: $MODEL/config.json 不存在" >&2; exit 1; }

# ---- 可调项（默认 = 已验证值）--------------------------------------------
PORT="${PORT:-8000}"
NAME="${NAME:-dsv41}"
SERVED_NAME="${SERVED_NAME:-DeepSeek-V4.1-Flash}"
TP="${TP:-8}"
MAXLEN="${MAXLEN:-1048576}"
BAT="${BAT:-4096}"
MAX_SEQS="${MAX_SEQS:-32}"
GPU_UTIL="${GPU_UTIL:-0.92}"
BLOCK="${BLOCK:-128}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
SRC_DIR="${SRC_DIR:-$HERE/tree}"          # 打补丁后的源码树放这里
PGO_SO="${PGO_SO:-}"                      # 可选：PGO 版 libpython（见 README §5）
LOG_DIR="${LOG_DIR:-$HERE/runtime/$(hostname -s)/$NAME}"
IMAGE="$IMAGE"; export IMAGE MODEL

# ---- KV 外部池（可选；KV_OFFLOAD=1 才启用）--------------------------------
# 实现不在本文件，在 kv_offload.sh —— 与工作区根目录 start-server.sh 共用同一份，
# 免得像 GPU_UTIL 那回一样两处各写一遍再漂移。变量默认值也都在那里。
# 关掉时脚本行为与从前逐字一致：不建容器、不写 conf、不加任何 vllm 参数。
KV_CONF_DIR="${KV_CONF_DIR:-$LOG_DIR/kv}"   # 实例级 conf 目录，交给 kv_offload.sh
source "$HERE/kv_offload.sh"

# decode 桶表：上游 27 桶 + 补 7 个 6N（draft 一步吃 num_reqs×(1+SPEC_TOKENS) 个 token，
# 缺 6N 时 ACL graph 补位会把 draft 打回 eager，实测每输出时延 +26%）。
CUDAGRAPH_SIZES="${CUDAGRAPH_SIZES:-1,2,4,8,16,24,30,32,40,48,54,56,64,72,78,80,88,96,102,104,112,120,126,128,136,144,150,152,160,168,174,176,184,192}"

# 6 个开关，**全部是"静默失效"型**：不传服务照常起，只是白丢收益，日志不报错。
#
# 为什么是 6 个而不是 V1 分支的 12 个：另外 6 个 VLLM_ASCEND_DSPARK_* 只在
# spec_decode/dspark_proposer.py 与 llm_base_proposer.py 里被读，而这两个文件
# 属于 V1 的 proposer 路径。V2 走 worker/v2/spec_decode/dspark/speculator.py
# （继承上游 DSparkSpeculator + DFlashAclGraphManager），根本不 import 它们。
# V1 里那 6 个开关换来的「draft 进 ACLGraph」，在 V2 由 query_cudagraph_manager
# 原生提供。传了不报错也不生效，所以本分支既不定义也不传。
PASSTHRU_ENVS_STR="${PASSTHRU_ENVS_STR:-\
-e VLLM_ASCEND_V41_TRITON_SLOT_MAPPING=1 \
-e VLLM_ASCEND_ROPE_INDEX_SELECT=1 \
-e VLLM_ASCEND_V41_MOE_MASK_RANGE=1 \
-e VLLM_ASCEND_V41_O_PROJ_2D=1 \
-e VLLM_ASCEND_V41_QLI_NO_CANDIDATE=1 \
-e VLLM_ASCEND_V41_ENGRAM_WKV_TP=1}"

for t in docker tar; do
  command -v "$t" >/dev/null 2>&1 || { echo "FATAL: 缺命令 $t" >&2; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FATAL: 本机没有镜像 $IMAGE（先 docker pull / docker load）" >&2; exit 1; }

mkdir -p "$LOG_DIR/logs/ascend_plog" "$LOG_DIR/logs/ascend_npulog" \
         "$LOG_DIR/run" "$LOG_DIR/cache/vllm_cache" "$LOG_DIR/cache/kernel_meta" \
         "$LOG_DIR/cache/static_kernel_compile_outputs"
LOG="$LOG_DIR/logs/server-$(date '+%Y%m%d-%H%M%S').log"

# ---- 已有一个吃 8 卡的容器就别重复起 -------------------------------------
# 只查 $NAME 是不够的：本机可能还跑着**另一个名字**的 8 卡容器（实测：工作区
# 那边是 dsv41-rc2，本脚本默认 NAME=dsv41）。那样 FORCE=1 停不掉它，于是两个
# 容器同时抢 8 张卡和 8000 端口 —— 起服会以各种奇怪方式失败（HBM 不够 /
# 端口占用 / HCCL 初始化超时），而且要好几分钟才发现。
#
# ALLOW_FOREIGN：本机常驻、与本服务无争用的容器白名单（空格分隔）。
#   deepseek-v4 是 2026-08 起的陈旧容器，只占着设备节点、不做推理（AGENTS.md
#   记为"与本工作区无关，别去动"），所以默认放行。真要连它一起管就置空。
ALLOW_FOREIGN="${ALLOW_FOREIGN-deepseek-v4}"
if [ -n "$(docker ps -q -f "name=^${NAME}$" 2>/dev/null || true)" ]; then
  if [ "${FORCE:-0}" = "1" ]; then
    echo "[start] FORCE=1 -> docker stop -t 30 $NAME"; docker stop -t 30 "$NAME" >/dev/null
  else
    echo "[start] 容器 $NAME 已在运行；要重启：FORCE=1 bash $0" >&2; exit 1
  fi
fi

# 本机上除了 $NAME 和白名单，还有没有别的容器占着 davinci 设备？
foreign_host=""
for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
  [ "$c" = "$NAME" ] && continue
  case " $ALLOW_FOREIGN " in *" $c "*) continue ;; esac
  if docker inspect -f '{{json .HostConfig.Devices}}' "$c" 2>/dev/null | grep -q '/dev/davinci0'; then
    foreign_host="$foreign_host $c"
  fi
done
if [ -n "$foreign_host" ]; then
  echo "FATAL: 本机还有别的容器占着 8 张卡:$foreign_host" >&2
  echo "       FORCE=1 只会停 $NAME，停不掉它们。同机只能有一个吃 8 卡的容器。" >&2
  echo "       先停掉：docker stop$foreign_host" >&2
  echo "       若其中某个确实与本服务无争用，加白名单：" >&2
  echo "         ALLOW_FOREIGN=\"\$ALLOW_FOREIGN <名字>\" bash $0" >&2
  exit 2
fi
# 端口：走到这里说明 $NAME 不在跑，那么 $PORT 上如果有人在听就是外来的
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  echo "FATAL: 端口 ${PORT} 已被占用（且 $NAME 未在运行）" >&2
  echo "       查：ss -ltnp | grep :${PORT}" >&2
  echo "       换端口：PORT=8001 bash $0" >&2
  exit 2
fi

# ---- 源码树：首次从镜像里取出 stock 树，再打两个补丁 ----------------------
# 为什么要这一步：v25 的改动全在 Python 侧，但运行必需的 .so（C 扩展、CANN 算子、
# rust parser）在镜像里、且被 .gitignore 排除 ⇒ 打补丁的树必须**基于镜像自带的树**
# 才配对。从镜像里 tar 出来（排除编译中间产物）再 apply，大小约 0.6 GB。
# 序列指纹 = 基础镜像身份 + series.conf + 每个 patch 的内容哈希。
#
# ★ 镜像身份必须进指纹。镜像自带的树是"补丁的基版"，换镜像却不重建树 =
#   拿 0.30.0rc2 的代码去跑 0.31.0rc1（静默跑旧码）。旧实现只哈希 series.conf
#   与补丁，换 IMAGE 后指纹不变 ⇒ 静默复用旧树。
# 旧实现还只看 .patches-applied **文件是否存在**，改了补丁却不重建树 ⇒ 静默用旧树。
# 真实事故：v2 把补丁从 1788 行裁到 796 行，tree/ 里仍是 v1 的改动（941 行），
# 脚本照用不误。
image_identity() {
  docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || echo "UNKNOWN-IMAGE"
}

series_fingerprint() {
  local list
  list=$(sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$HERE/series.conf")
  {
    printf 'image=%s\n' "$(image_identity)"
    printf '%s\n' "$list"
    local f
    for f in $list; do
      if [ -f "$HERE/patches/$f" ]; then
        md5sum "$HERE/patches/$f"
      else
        echo "MISSING $f"
      fi
    done
    md5sum "$HERE/vllm.patch" 2>/dev/null
  } | md5sum | cut -d' ' -f1
}

# 树的**实际**状态指纹：两棵树都是 git 仓库，让 git 算内容哈希。
# 比逐文件 md5 更准（能区分 改/新增/删除）也更省（0.35 s vs 3.8 s / 17.8k 文件）。
# 镜像里的树是 pristine（status 为空），所以打补丁后 status 恰好等于"补丁触及
# 的文件集"；任何**额外**的条目都说明树被手改过或带了残留。
# 真实事故：tree 里有 tests/ut/attention/test_dsa_v41_metadata_reuse.py，
# 镜像没有、任何补丁都不带，却静静躺在树里（?? 行）两个月没人发现。
tree_state_fingerprint() {
  local d
  for d in "$SRC_DIR/vllm-ascend" "$SRC_DIR/vllm"; do
    printf '== %s\n' "${d/#$SRC_DIR/}"      # 只留相对名，SRC_DIR 变了不算树变了
    if [ -e "$d/.git" ]; then
      git -C "$d" status --porcelain 2>/dev/null || echo "GIT-ERROR"
      # ⚠ 必须连**内容**一起哈希，不能只看 status：status 只说"哪个文件被改了"，
      #   不说"改成了什么"。补丁本来就改了那 9 个文件，再改一次它们的内容，
      #   status 一个字都不变（实测：改 dsa_v41.py 后 status 指纹仍为 90f94c51）。
      git -C "$d" diff 2>/dev/null || echo "GIT-DIFF-ERROR"
    else
      echo "NO-GIT"
    fi
  done | md5sum | cut -d' ' -f1
}

WANT_FP=$(series_fingerprint)
# 注意：脚本是 set -euo pipefail。`cat missing | head` 在 pipefail 下返回非 0
# 会让赋值失败并静默退出，所以用 `|| true` 兜住"标记文件不存在"这个正常情况。
HAVE_FP=$(sed -n '1p' "$SRC_DIR/.patches-applied" 2>/dev/null || true)
HAVE_STATE=$(sed -n '2p' "$SRC_DIR/.patches-applied" 2>/dev/null || true)

# 要么补丁/镜像变了（指纹不同），要么树的内容与"刚建好时"不一致（被手改/有残留）。
need_prepare=0
prepare_why=""
if [ "$WANT_FP" != "$HAVE_FP" ]; then
  need_prepare=1; prepare_why="补丁序列或基础镜像变了（期望 ${WANT_FP:0:12}，树上是 ${HAVE_FP:0:12}）"
elif [ -z "$HAVE_STATE" ]; then
  need_prepare=1; prepare_why="树里没有状态指纹（旧版本建的树，无法确认内容）"
else
  actual_state=$(tree_state_fingerprint)
  if [ "$actual_state" != "$HAVE_STATE" ]; then
    need_prepare=1
    prepare_why="树的内容与刚建好时不一致（被改过，或混进了不该有的文件）"
  fi
fi

if [ "${SKIP_PREPARE:-0}" = "1" ] && [ "$need_prepare" = "1" ]; then
  echo "[start] SKIP_PREPARE=1：跳过重建，但树的状态不对 —— $prepare_why" >&2
  echo "[start]   （你在手工调树的话这正是预期；否则去掉 SKIP_PREPARE 让它重建）" >&2
  need_prepare=0
fi

# ★ DRY_RUN 必须在**任何副作用之前**生效。这个脚本的 DRY_RUN 原本只拦截最后的
#   docker run，前面的 rm -rf / tar 解包 / 打补丁照样执行 —— 2026-10-28 实测把
#   tree/vllm-ascend 清空了（rm 因容器占着 .nfs* 文件而报错，其余全删掉）。
#   教训：凡是"只打印"的开关，必须在第一个写操作之前生效，不能只拦最后一步。
#   这里不直接退出 —— DRY_RUN 的用途就是预览 docker 命令，跳过后面的树相关校验
#   （DRY_RUN_SKIP_TREE=1）照样把它打印出来。
DRY_RUN_SKIP_TREE=0
if [ "${DRY_RUN:-0}" = "1" ] && [ "$need_prepare" = "1" ]; then
  echo "[start] DRY_RUN=1：树需要重建 —— $prepare_why" >&2
  echo "[start]   DRY_RUN 不动磁盘，跳过重建与树相关校验；下面打印的命令基于当前树。" >&2
  need_prepare=0
  DRY_RUN_SKIP_TREE=1
fi

if [ "$need_prepare" = "1" ]; then
  if [ -n "$HAVE_FP" ]; then
    echo "[start] 重建源码树 —— $prepare_why"
    echo "[start]   旧树里若有手改内容会被覆盖（tree/ 是产物，不是工作区）"
  else
    echo "[start] 准备源码树 -> $SRC_DIR（首次约 1 min，约 0.6 GB）"
  fi
  # 重建前必须清空两个子树：tar 解包是**覆盖**，不会删除旧树里多出来的文件。
  # 实测残留就是这么来的：tree 里曾有 tests/ut/attention/test_dsa_v41_metadata_reuse.py
  # （镜像没有、任何补丁都不带），一路跟着每次"重建"活了下来。
  # 删除前确认目标确实在 SRC_DIR 之下，且 SRC_DIR 是绝对路径且不像 / 。
  case "$SRC_DIR" in
    /|/mnt|/mnt/nfs|/mnt/nfs/services) echo "FATAL: 拒绝在 $SRC_DIR 下删树，SRC_DIR 太浅" >&2; exit 2 ;;
    /*) ;;
    *) echo "FATAL: SRC_DIR=$SRC_DIR 必须是绝对路径（重建要删旧树）" >&2; exit 2 ;;
  esac
  for sub in vllm-ascend vllm; do
    target="$SRC_DIR/$sub"
    case "$target" in
      "$SRC_DIR"/*) ;;
      *) echo "FATAL: 拒绝删除 $target（不在 $SRC_DIR 之下）" >&2; exit 2 ;;
    esac
    [ -e "$target" ] && { echo "[start]   清空旧 $sub"; rm -rf -- "$target"; }
  done
  mkdir -p "$SRC_DIR/vllm-ascend" "$SRC_DIR/vllm"
  docker run --rm --entrypoint tar "$IMAGE" -C /vllm-workspace -cf - \
      --exclude='vllm-ascend/csrc/build' --exclude='vllm-ascend/csrc/build_cache' \
      --exclude='vllm-ascend/csrc/build_out' --exclude='vllm-ascend/csrc/output' \
      --exclude='vllm/rust/target' vllm-ascend vllm \
    | tar -xf - -C "$SRC_DIR"

  [ -f "$HERE/vllm.patch" ] || { echo "FATAL: 缺补丁 $HERE/vllm.patch" >&2; exit 1; }
  # 按 series.conf 顺序逐个应用（分层：0xxx 优化 / 1xxx KV offload）
  while read -r pf; do
    case "$pf" in ''|'#'*) continue ;; esac
    [ -f "$HERE/patches/$pf" ] || { echo "FATAL: series 里的 $pf 不存在" >&2; exit 1; }
    if command -v git >/dev/null 2>&1; then
      git -C "$SRC_DIR/vllm-ascend" apply "$HERE/patches/$pf" \
        || { echo "FATAL: git apply 失败 $pf（用 DRY_RUN=1 或手工检查）" >&2; exit 1; }
    else
      patch -p1 -d "$SRC_DIR/vllm-ascend" < "$HERE/patches/$pf" \
        || { echo "FATAL: patch 失败 $pf" >&2; exit 1; }
    fi
    echo "[start]   applied $pf"
  done < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$HERE/series.conf")

  if command -v git >/dev/null 2>&1; then
    git -C "$SRC_DIR/vllm" apply "$HERE/vllm.patch" || { echo "FATAL: git apply 失败 vllm.patch" >&2; exit 1; }
  else
    patch -p1 -d "$SRC_DIR/vllm" < "$HERE/vllm.patch" || { echo "FATAL: patch 失败 vllm.patch" >&2; exit 1; }
  fi
  # 第 1 行 = 序列+镜像指纹（决定"该不该重建"）
  # 第 2 行 = 建好那一刻的 git 状态指纹（决定"建好之后有没有被动过"）
  # 第 3 行 = 建树用的镜像 Id（**明文**）：别的入口（工作区根 start-server.sh）要判断
  #          "这棵树是不是用本次要挂的镜像建的"。指纹第 1 行是哈希，外部拿不到镜像
  #          身份、只能自己再算一遍 —— 那就会像 GPU_UTIL 那回一样两处各写一遍再漂移。
  # 两行都要，只有第 1 行就发现不了"补丁没变但树被手改/混进残留"。
  {
    printf '%s\n' "$WANT_FP"
    tree_state_fingerprint
    printf '%s\n' "$(image_identity)"
  } > "$SRC_DIR/.patches-applied"
  echo "[start] 补丁已应用（fingerprint ${WANT_FP:0:12}，树状态 $(tree_state_fingerprint | cut -c1-12)）"
fi

# 构建产物硬校验：worktree / 复制树缺了直接 ImportError，别等 9 min 后才发现
# DRY_RUN 且树待重建时跳过（此时树还是旧的/空的，校验只会掩盖真正要打印的命令）
if [ "${DRY_RUN_SKIP_TREE:-0}" != "1" ]; then
for f in "$SRC_DIR/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so" \
         "$SRC_DIR/vllm-ascend/vllm_ascend/libvllm_ascend_kernels.so" \
         "$SRC_DIR/vllm/vllm/_rust_tool_parser.abi3.so"; do
  [ -f "$f" ] || { echo "FATAL: 缺构建产物 $f（应从镜像里取出，见 README §3）" >&2; exit 1; }
done
# 12 个 gate 必须都在**将要挂载那棵树**的 envs.py 里有定义，否则 vLLM 只打一行
# "Unknown vLLM environment variable"，服务照常起、收益全丢。这里前置拦。
missing=""
for g in $(printf '%s\n' "$PASSTHRU_ENVS_STR" | grep -oE 'VLLM_ASCEND_[A-Z0-9_]+=1' | sed 's/=1$//'); do
  grep -q "\"$g\"" "$SRC_DIR/vllm-ascend/vllm_ascend/envs.py" 2>/dev/null || missing="$missing $g"
done
if [ -n "$missing" ]; then
  echo "FATAL: 这些 gate 传了 =1，但 envs.py 里没有定义:$missing" >&2
  echo "       补丁没打成功？或 SRC_DIR 指向了未打补丁的树。" >&2
  exit 3
fi
fi   # DRY_RUN_SKIP_TREE

PGO_MOUNTS=()
if [ -n "$PGO_SO" ]; then
  [ -f "$PGO_SO" ] || { echo "FATAL: PGO_SO=$PGO_SO 不存在" >&2; exit 1; }
  PGO_MOUNTS=(-v "$PGO_SO:/usr/local/python3.12.13/lib/libpython3.12.so.1.0:ro")
  echo "[start] pgo=1 $(md5sum "$PGO_SO" | cut -c1-12)"
fi

# ---- KV 外部池（KV_OFFLOAD=1 才启用；实现在 kv_offload.sh）----------------
# 两件事：① 按挂载树校验 0101（正确性闸门，理由见 kv_offload.sh 里的注释）；
# ② 写 conf + 起/复用 MetaService，填出 KV_ENVS/KV_MOUNTS/KV_ARGS 三个数组。
kv_offload_require_tree_fix "$SRC_DIR/vllm-ascend"
kv_offload_setup
export KV_OFFLOAD

read -r -a PASSTHRU_ENVS <<< "$PASSTHRU_ENVS_STR"

COMPILATION_JSON="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[${CUDAGRAPH_SIZES}]}"
ADDL_JSON='{"enable_cpu_binding":true,"ascend_compilation_config":{"enable_npugraph_ex":false,"enable_static_kernel":false}}'
SPEC_JSON="{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"enforce_eager\":false}"
MTLOAD_JSON='{"enable_multithread_load":true,"num_threads":128}'

RUN_ARGS=(
  --name "$NAME" --privileged --network host --shm-size 512g
  --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3
  --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc
  -e ASCEND_RT_VISIBLE_DEVICES="${DEVICES_SPEC:-0,1,2,3,4,5,6,7}"
  -e PYTHONPYCACHEPREFIX=/workspace/run/pycache
  -e TRITON_ASCEND_ARCH="${TRITON_ARCH:-Ascend910B1}"
  -e VLLM_USE_V2_MODEL_RUNNER=1
  -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
  -e TASK_QUEUE_ENABLE=1
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
  -e LOCAL_WORLD_SIZE="$TP"
  -e LD_PRELOAD="${JEMALLOC:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
  "${PASSTHRU_ENVS[@]}"
  "${KV_ENVS[@]}"
  "${PGO_MOUNTS[@]}"
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v /tmp:/tmp
  "${KV_MOUNTS[@]}"
  -v "$MODEL:$MODEL:ro"
  -v "$LOG_DIR/run:/workspace/run:rw"
  -v "$LOG_DIR/logs:/workspace/logs:rw"
  -v "$LOG_DIR/logs/ascend_plog:/root/ascend:rw"
  -v "$LOG_DIR/logs/ascend_npulog:/var/log/npu:rw"
  -v "$LOG_DIR/cache/static_kernel_compile_outputs:/workspace/static_kernel_compile_outputs:rw"
  -v "$LOG_DIR/cache/kernel_meta:/workspace/kernel_meta:rw"
  -v "$LOG_DIR/cache/vllm_cache:/root/.cache/vllm:rw"
  -v "$SRC_DIR/vllm-ascend:/vllm-workspace/vllm-ascend:${SRC_MOUNT_MODE:-ro}"
  -v "$SRC_DIR/vllm:/vllm-workspace/vllm:${SRC_MOUNT_MODE:-ro}"
  -w /workspace --entrypoint vllm "$IMAGE"
  serve "$MODEL"
  --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT"
  --tensor-parallel-size "$TP" --enable-expert-parallel
  --max-model-len "$MAXLEN" --max-num-batched-tokens "$BAT" --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "$GPU_UTIL" --block-size "$BLOCK"
  --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --trust-remote-code
  --quantization ascend --safetensors-load-strategy lazy
  --compilation-config "$COMPILATION_JSON"
  --model-loader-extra-config "$MTLOAD_JSON"
  --additional-config "$ADDL_JSON"
  --speculative-config "$SPEC_JSON"
  --enable-prompt-tokens-details
  "${KV_ARGS[@]}"
  "$@"     # 透传额外 vllm serve 参数，例如 KV offload：
           #   bash start-server.sh \
           #     --kv-transfer-config '{"kv_connector":"AscendStoreConnector",...}'
)

if [ "${DRY_RUN:-0}" = "1" ]; then
  printf 'docker run'; printf ' %q' "${RUN_ARGS[@]}"; echo
  echo "[start] DRY_RUN=1，未启动。log 将写入 $LOG"
  exit 0
fi

{
  echo "=============================================================="
  echo "[start] $(date '+%F %T %Z') host=$(hostname -s) image=$IMAGE model=$MODEL"
  echo "[start] bat=$BAT maxlen=$MAXLEN max_seqs=$MAX_SEQS util=$GPU_UTIL spec=$SPEC_TOKENS block=$BLOCK"
  echo "[start] src=$SRC_DIR"
  # 用 if/else 而不是 `[ ... ] && echo`：后者在本块里虽然不会被 set -e 打断
  # （&& 链里非末位的命令被豁免），但读起来像"可能静默退出"，改成显式分支。
  if [ "$KV_OFFLOAD" = "1" ]; then
    echo "[start] kv_offload=1 ${KV_DRAM_GB}GB/rank $KV_PROTOCOL meta=$KV_META_NAME"
  else
    echo "[start] kv_offload=0"
  fi
} > "$LOG"

echo "[start] 起服中（8 卡图捕获，实测约 9~11 min 到 READY）"
echo "[start] log=$LOG"
echo "[start] 跟踪：tail -f $LOG"
nohup docker run --rm "${RUN_ARGS[@]}" >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/server.pid"

# ---- WAIT=1：等 READY，并硬校验 4 件"静默失效"的东西 ----------------------
if [ "${WAIT:-0}" = "1" ]; then
  # 宽限期不能省：上面是 nohup 后台 docker run 并立即返回，此时容器还没创建出来，
  # `docker ps` 查不到 —— 直接判"已退出"会把正常启动误报成失败（实测踩过）。
  T=0; GRACE=120; st=""
  while :; do
    st="$(docker ps -a --filter "name=^${NAME}$" --format '{{.Status}}' | head -1)"
    if [ -z "$st" ]; then
      (( T < GRACE )) || { echo "[start] FATAL: ${GRACE}s 内未见容器 $NAME，看 $LOG" >&2; exit 3; }
      sleep 5; T=$((T+5)); continue
    fi
    echo "$st" | grep -q '^Up' || { echo "[start] FATAL: 容器已退出（$st），看日志 $LOG" >&2; exit 3; }
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && break
    (( T >= ${READY_TIMEOUT:-2700} )) && { echo "[start] FATAL: 超时未 READY" >&2; exit 1; }
    sleep 10; T=$((T+10))
    if (( T % 60 == 0 )); then echo "[start] ${T}s ..."; fi
  done
  echo "[start] READY after ${T}s"

  # V2 的判据与 V1 完全不同：V1 看 draft 进图（runtime_mode=FULL / [dspark-graph-capture]），
  # 那两条只在 V1 的 proposer 里打印，V2 下恒为 0 —— 照抄 V1 会把成功判成失败。
  n_v2=$(grep -c 'npu model runner v2 is in developing' "$LOG" || true)
  n_v1mark=$(grep -c 'runtime_mode=FULL\|\[dspark-graph-capture\]' "$LOG" || true)
  n_tgt=$(tr '\r' '\n' < "$LOG" | grep -oE 'Capturing CUDA graphs \(FULL\): +[0-9]+%[^|]*\| *[0-9]+/[0-9]+' | grep -oE '[0-9]+/[0-9]+$' | cut -d/ -f1 | sort -un | paste -sd, -)
  echo "[start] 校验：V2 worker 标记 ${n_v2}/${TP}，目标图 ${n_tgt:-<未找到>}"
  echo "[start]       V1-only 标记应为 0：${n_v1mark}"
  grep -oE 'NPU KV cache size: [0-9,]+ tokens' "$LOG" | tail -1 || true
  [ "$n_v2" -ge 1 ] || {
    echo "[start] FATAL: 日志里没有 V2 worker 标记，实际走的可能还是 V1" >&2
    echo "[start]   查：docker exec $NAME env | grep V2_MODEL_RUNNER" >&2; exit 4; }
  [ "$n_v2" -ge "$TP" ] || echo "[start] 警告：V2 标记只有 ${n_v2} 条，期望 ${TP}" >&2
  [ "$n_v1mark" -eq 0 ] || echo "[start] 警告：出现 ${n_v1mark} 条 V1 专属日志行" >&2

  # KV 池也是"静默失效"型：connector 名字打出来了但池没连上，服务照样起、
  # 只是每次都没命中，收益全丢。三条都要过才是真生效。
  if [ "$KV_OFFLOAD" = "1" ]; then
    n_conn=$(grep -c 'Creating v1 connector with name: AscendStoreConnector' "$LOG" || true)
    n_reg=$(grep -c 'RegisterBaseInfo.*register capacity' "$LOG" || true)
    d_hit=$(grep -c 'use_hybrid: True' "$LOG" || true)
    cap=$(curl -sf --max-time 3 "http://127.0.0.1:${KV_METRICS_PORT}/metrics" 2>/dev/null \
          | awk -F'[{} ]' '/^memcache_total_capacity_bytes/{s+=$NF} END{printf "%.1f", s/1073741824}')
    echo "[start] 校验：AscendStore connector ${n_conn}/${TP}，rank 登记 ${n_reg}，池容量 ${cap:-?} GiB"
    [ "$n_conn" -ge "$TP" ] || { echo "[start] FATAL: connector 只建了 ${n_conn} 个，期望 ${TP}" >&2; exit 7; }
    [ "$n_reg" -ge "$TP" ] || { echo "[start] FATAL: 只有 ${n_reg} 个 rank 登记到 meta，池没真正接上" >&2; exit 7; }
    [ "$d_hit" -ge "$TP" ] || echo "[start] 警告：use_hybrid 行只有 ${d_hit} 条，期望 ${TP}" >&2
  fi

  # prompt_tokens_details 也是静默失效型：没生效时 usage 里该字段恒为 null
  curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" \
    | grep -q prompt_tokens_details \
    && echo "[start] 校验：usage 里有 prompt_tokens_details" \
    || echo "[start] 警告：usage 里没有 prompt_tokens_details" >&2
fi
