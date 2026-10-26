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
if [ -n "$(docker ps -q -f "name=^${NAME}$" 2>/dev/null || true)" ]; then
  if [ "${FORCE:-0}" = "1" ]; then
    echo "[start] FORCE=1 -> docker stop -t 30 $NAME"; docker stop -t 30 "$NAME" >/dev/null
  else
    echo "[start] 容器 $NAME 已在运行；要重启：FORCE=1 bash $0" >&2; exit 1
  fi
fi

# ---- 源码树：首次从镜像里取出 stock 树，再打两个补丁 ----------------------
# 为什么要这一步：v25 的改动全在 Python 侧，但运行必需的 .so（C 扩展、CANN 算子、
# rust parser）在镜像里、且被 .gitignore 排除 ⇒ 打补丁的树必须**基于镜像自带的树**
# 才配对。从镜像里 tar 出来（排除编译中间产物）再 apply，大小约 0.6 GB。
if [ "${SKIP_PREPARE:-0}" != "1" ] && [ ! -f "$SRC_DIR/.patches-applied" ]; then
  echo "[start] 准备源码树 -> $SRC_DIR（首次约 1 min，约 0.6 GB）"
  mkdir -p "$SRC_DIR/vllm-ascend" "$SRC_DIR/vllm"
  docker run --rm --entrypoint tar "$IMAGE" -C /vllm-workspace -cf - \
      --exclude='vllm-ascend/csrc/build' --exclude='vllm-ascend/csrc/build_cache' \
      --exclude='vllm-ascend/csrc/build_out' --exclude='vllm-ascend/csrc/output' \
      --exclude='vllm/rust/target' vllm-ascend vllm \
    | tar -xf - -C "$SRC_DIR"

  for p in vllm-ascend.patch vllm.patch; do
    [ -f "$HERE/$p" ] || { echo "FATAL: 缺补丁 $HERE/$p" >&2; exit 1; }
  done
  if command -v git >/dev/null 2>&1; then
    git -C "$SRC_DIR/vllm-ascend" apply "$HERE/vllm-ascend.patch"
    git -C "$SRC_DIR/vllm"        apply "$HERE/vllm.patch"
  else
    patch -p1 -d "$SRC_DIR/vllm-ascend" < "$HERE/vllm-ascend.patch"
    patch -p1 -d "$SRC_DIR/vllm"        < "$HERE/vllm.patch"
  fi
  md5sum "$HERE"/*.patch > "$SRC_DIR/.patches-applied"
  echo "[start] 补丁已应用"
fi

# 构建产物硬校验：worktree / 复制树缺了直接 ImportError，别等 9 min 后才发现
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

PGO_MOUNTS=()
if [ -n "$PGO_SO" ]; then
  [ -f "$PGO_SO" ] || { echo "FATAL: PGO_SO=$PGO_SO 不存在" >&2; exit 1; }
  PGO_MOUNTS=(-v "$PGO_SO:/usr/local/python3.12.13/lib/libpython3.12.so.1.0:ro")
  echo "[start] pgo=1 $(md5sum "$PGO_SO" | cut -c1-12)"
fi

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
  "${PGO_MOUNTS[@]}"
  -v /usr/local/dcmi:/usr/local/dcmi:ro
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
  -v /etc/hccn.conf:/etc/hccn.conf:ro
  -v /tmp:/tmp
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

  # prompt_tokens_details 也是静默失效型：没生效时 usage 里该字段恒为 null
  curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" \
    | grep -q prompt_tokens_details \
    && echo "[start] 校验：usage 里有 prompt_tokens_details" \
    || echo "[start] 警告：usage 里没有 prompt_tokens_details" >&2
fi
