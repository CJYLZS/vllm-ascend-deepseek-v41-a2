#!/usr/bin/env bash
# =============================================================================
# kv_offload.sh —— KV 外部池（AscendStoreConnector + MemCache）的唯一实现
#
# 被两个入口 source，实现只有这一份：
#   ① 本仓库的 start-server.sh      —— 自包含入口（自带 tree/，README §6/§7）
#   ② 工作区根目录 start-server.sh  —— 挂 rc2/stable 的那条入口
# 这样"起池"这件事只有一个实现，不会像 GPU_UTIL 那回一样两处各写一遍再漂移。
#
# 契约（调用方准备；模块只在缺省时兜底）：
#   NAME          容器名基名（meta 容器 = ${NAME}-mmc-meta）
#   TP            每机卡数，用来把 dram.size（★ 每 rank）换算成整机容量
#   IMAGE         起 meta 容器用的镜像（KV_OFFLOAD=1 时必需）
#   KV_CONF_DIR   实例级 conf 目录，必须绝对路径（如 <runtime>/kv）
#   DRY_RUN=1     只打印将做什么：不启容器、不 curl
#
# 产出（调用方拼 docker run 用）：
#   KV_ACTIVE=0|1
#   KV_ENVS[]  KV_MOUNTS[]  KV_ARGS[]
#   KV_META_NAME / KV_META_PORT / KV_STORE_PORT / KV_METRICS_PORT
#
# 用法：
#   source kv_offload.sh
#   kv_offload_plan             # 无副作用：校验 + 写 conf + 填三个数组
#   kv_offload_ensure_meta      # 有副作用：复用或启动 MetaService，并等它就绪
#   kv_offload_setup            # = plan + ensure_meta（一次性调用）
#   kv_offload_require_tree_fix <挂载树根|"">   # 校验树里有没有 0101
#
# 三件事缺一不可，缺任何一件 vllm 都会在 engine 初始化时 assert 死：
#   ① MetaService 容器（默认 5100/6100/8100，不占卡、不吃 HBM）
#   ② 一份 mmc-local.conf，路径经 MMC_LOCAL_CONFIG_PATH 传给引擎
#   ③ --kv-transfer-config '{"kv_connector":"AscendStoreConnector",...}'
# =============================================================================

KV_OFFLOAD="${KV_OFFLOAD:-0}"              # 1 = 挂 AscendStoreConnector + memcache
KV_DRAM_GB="${KV_DRAM_GB:-2}"              # ★ 每 rank 的 DRAM 池，不是整机
KV_PROTOCOL="${KV_PROTOCOL:-host_shm}"     # 见 validate 里的实测说明，别随便改
KV_WORLD_SIZE="${KV_WORLD_SIZE:-256}"      # 可登记的总 rank 上限；改了要重启 meta
KV_META_NAME="${KV_META_NAME:-${NAME:-dsv41-rc2}-mmc-meta}"
KV_META_PORT="${KV_META_PORT:-5100}"
KV_STORE_PORT="${KV_STORE_PORT:-6100}"
KV_METRICS_PORT="${KV_METRICS_PORT:-8100}" # 不能是 8000：会和 vllm 的 --port 撞
KV_CONF_DIR="${KV_CONF_DIR:-}"             # 由调用方给（实例级）
KV_CONF_IN_CT="${KV_CONF_IN_CT:-/workspace/kvconf}"
# 逃生阀：挂载树缺 0101 修复时默认 FATAL；确实要带病跑才置 1。
KV_ALLOW_UNFIXED_TREE="${KV_ALLOW_UNFIXED_TREE:-0}"

KV_ACTIVE=0
KV_ENVS=()
KV_MOUNTS=()
KV_ARGS=()

kv_offload_plan() {
  KV_ACTIVE=0
  KV_ENVS=(); KV_MOUNTS=(); KV_ARGS=()

  case "$KV_OFFLOAD" in
    0|1) ;;
    *) echo "FATAL: KV_OFFLOAD=$KV_OFFLOAD（应为 0/1）" >&2; exit 2 ;;
  esac
  [ "$KV_OFFLOAD" = "1" ] || return 0

  local t
  for t in docker curl; do
    command -v "$t" >/dev/null 2>&1 || { echo "FATAL: KV_OFFLOAD=1 但缺命令 $t" >&2; exit 1; }
  done
  [ -n "${IMAGE:-}" ] || { echo "FATAL: KV_OFFLOAD=1 需要 IMAGE（起 meta 容器要用）" >&2; exit 1; }
  case "$KV_CONF_DIR" in
    "")      echo "FATAL: KV_OFFLOAD=1 但 KV_CONF_DIR 未设置（应由调用方给实例级绝对路径）" >&2; exit 2 ;;
    /*)      ;;
    *)       echo "FATAL: KV_CONF_DIR=$KV_CONF_DIR 必须是绝对路径" >&2; exit 2 ;;
  esac

  # dram.size 允许小数（实测 1.5GB 可用），所以不用整数算术校验
  if ! printf '%s' "$KV_DRAM_GB" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    echo "FATAL: KV_DRAM_GB=$KV_DRAM_GB（应为正数，单位 GB，★ 每 rank）" >&2; exit 2
  fi
  if ! printf '%s' "$KV_WORLD_SIZE" | grep -Eq '^[0-9]+$'; then
    echo "FATAL: KV_WORLD_SIZE=$KV_WORLD_SIZE（应为正整数）" >&2; exit 2
  fi

  # 端口：任何一个和 vllm 的 --port 撞都会在起服 8 分钟后以难懂的方式失败
  local p
  for p in "$KV_META_PORT" "$KV_STORE_PORT" "$KV_METRICS_PORT"; do
    case "$p" in ''|*[!0-9]*) echo "FATAL: KV 端口非法：$p" >&2; exit 2 ;; esac
    if [ -n "${PORT:-}" ] && [ "$p" = "$PORT" ]; then
      echo "FATAL: KV 端口 $p 与 vllm 的 --port 相同（改 KV_META_PORT/KV_STORE_PORT/KV_METRICS_PORT）" >&2
      exit 2
    fi
  done
  case "$KV_PROTOCOL" in
    host_shm) ;;
    host_rdma)
      echo "[kv] 警告：host_rdma 在本镜像上不可用 —— libmf_hybm_accoffload.so 是 0 字节，" >&2
      echo "[kv]       ExtendLibrary 加载失败（-6）。实测只有 host_shm 能 8/8 起来。" >&2 ;;
    device_rdma)
      echo "[kv] 警告：device_rdma 单 rank 能过、8 rank 必 segfault（RaSocketBatchConnect）。" >&2
      echo "[kv]       别用单 rank 的结论外推；默认值是 host_shm。" >&2 ;;
    *) echo "FATAL: 未知 KV_PROTOCOL=$KV_PROTOCOL（host_shm|host_rdma|device_rdma）" >&2; exit 2 ;;
  esac

  mkdir -p "$KV_CONF_DIR"
  # 引擎侧：dram.size 是**每 rank**，8 卡要 ×8 内存（8 卡 2GB ⇒ 整机 16GB）
  cat > "$KV_CONF_DIR/mmc-local.conf" <<EOF
ock.mmc.meta_service_url = tcp://127.0.0.1:${KV_META_PORT}
ock.mmc.local_service.config_store_url = tcp://127.0.0.1:${KV_STORE_PORT}
ock.mmc.log_level = info
ock.mmc.local_service.world_size = ${KV_WORLD_SIZE}
ock.mmc.local_service.protocol = ${KV_PROTOCOL}
ock.mmc.local_service.dram.size = ${KV_DRAM_GB}GB
EOF
  # MetaService 侧：metrics_url 供起服后核对池容量（memcache_total_capacity_bytes）
  cat > "$KV_CONF_DIR/mmc-meta.conf" <<EOF
ock.mmc.meta_service_url = tcp://127.0.0.1:${KV_META_PORT}
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:${KV_STORE_PORT}
ock.mmc.meta_service.metrics_url = http://127.0.0.1:${KV_METRICS_PORT}
ock.mmc.log_level = info
EOF

  KV_ACTIVE=1
  KV_ENVS=(-e MMC_LOCAL_CONFIG_PATH="$KV_CONF_IN_CT/mmc-local.conf")
  KV_MOUNTS=(-v "$KV_CONF_DIR:$KV_CONF_IN_CT:ro")
  KV_ARGS=(--kv-transfer-config
    "{\"kv_connector\":\"AscendStoreConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"backend\":\"memcache\"}}")

  local machine
  machine=$(awk -v a="$KV_DRAM_GB" -v b="${TP:-8}" 'BEGIN{printf "%.1f", a*b}')
  echo "[kv] kv_offload=1 protocol=$KV_PROTOCOL dram=${KV_DRAM_GB}GB/rank (整机 ${machine}GB，${TP:-8} rank)"
  echo "[kv]            conf=$KV_CONF_DIR/mmc-local.conf -> $KV_CONF_IN_CT/mmc-local.conf"
  echo "[kv]            meta=$KV_META_NAME :$KV_META_PORT/:$KV_STORE_PORT/:$KV_METRICS_PORT"
}

# 0101（= 上游 6197fec5c 的等价 2 行）只在挂池时有意义，但它决定**正确性**：
# MRV2 抢占恢复后不清 _preempted_req_ids，下一次 RUNNING delta 会走抢占分支、
# 用新 block 表覆盖已有的，把**错误的 block** 关联到池 key 上 —— 之后命中池
# 就是读到别人的 KV。日志不报错，所以必须前置拦。
kv_offload_require_tree_fix() {
  # 判据用 KV_OFFLOAD 而不是 KV_ACTIVE：这样调用方先校验再 plan、或先 plan 再
  # 校验都行，不会因为顺序不同静默跳过这道闸门（曾写成 KV_ACTIVE，先调即失效）。
  [ "$KV_OFFLOAD" = "1" ] || return 0
  local tree_root="${1-}" f="" why=""
  if [ -z "$tree_root" ]; then
    why="当前挂的是镜像自带的 stock 树（SRC=0），它不可能有 0101"
  elif [ ! -f "$tree_root/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py" ]; then
    why="找不到 $tree_root/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
  else
    f="$tree_root/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py"
    if grep -q '_preempted_req_ids\.discard(request\.req_id)' "$f"; then
      # start-server.sh 先校验一次、serve.sh 再兜底一次（两人都可能单独被调用），
      # 成功行只说一次；export 让子进程也看到这个标记。
      [ "${KV_TREE_FIX_VERIFIED:-0}" = "1" ] || echo "[kv] 挂载树含 0101 修复（MRV2 抢占恢复）"
      KV_TREE_FIX_VERIFIED=1; export KV_TREE_FIX_VERIFIED
      return 0
    fi
    why="挂载树 $tree_root 里 _process_new_request 缺 0101 那 2 行（stock rc2 就是这样）"
  fi
  if [ "$KV_ALLOW_UNFIXED_TREE" = "1" ]; then
    echo "[kv] 警告：$why" >&2
    echo "[kv]       KV_ALLOW_UNFIXED_TREE=1 放行；抢占发生后池里可能挂着错误的 block" >&2
    return 0
  fi
  {
    echo "FATAL: KV_OFFLOAD=1 但源码树不含 KV 池的 MRV2 修复"
    echo "       原因：$why"
    echo "       后果：请求被抢占后，池 key 会关联到错误的 block —— 命中池即读到别人的 KV，"
    echo "             输出错但日志不报错（这是正确性问题，不是性能问题）。"
    echo "       三种处理："
    echo "         ① 挂补丁仓库的树（已含 0101）："
    echo "              SRC_ROOT=<补丁仓库>/tree bash $0"
    echo "         ② 把补丁仓库的 patches/0101-kvpool-mrv2-resume.patch 固化进 rc2/stable（见 AGENTS.md §3.1）"
    echo "         ③ 明知有风险仍要跑：KV_ALLOW_UNFIXED_TREE=1 bash $0"
  } >&2
  exit 8
}

kv_offload_ensure_meta() {
  [ "${KV_ACTIVE:-0}" = "1" ] || return 0
  local running meta_conf meta_ports want_ports miss p

  running="$(docker inspect -f '{{.State.Running}}' "$KV_META_NAME" 2>/dev/null || true)"
  if [ "$running" = "true" ]; then
    # 复用前必须确认它监听的端口和刚写的 local conf 一致，否则引擎会连到
    # 一个不存在的端口 —— 那种失败发生在起服 8 分钟后，代价很大。
    # 配置路径可能在 .Config.Env，也可能写在 bash -c 的字符串里（手工起的容器
    # 就是后者），所以两处都找；都找不到就退化成"端口在听"这个弱判据。
    meta_conf=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$KV_META_NAME" 2>/dev/null \
                | sed -n 's/^MMC_META_CONFIG_PATH=//p' | head -1)
    [ -n "$meta_conf" ] || meta_conf=$(docker inspect -f '{{join .Config.Cmd " "}}' "$KV_META_NAME" 2>/dev/null \
                | tr ' ' '\n' | sed -n 's/^MMC_META_CONFIG_PATH=//p' | head -1)
    meta_ports=""
    if [ -n "$meta_conf" ]; then
      meta_ports=$(docker exec "$KV_META_NAME" cat "$meta_conf" 2>/dev/null \
                   | sed -n 's/.*tcp:\/\/127\.0\.0\.1:\([0-9]*\).*/\1/p;s#.*metrics_url.*:\([0-9]*\).*#\1#p' \
                   | sort -u | paste -sd, -)
    fi
    if [ -z "$meta_ports" ]; then
      miss=""
      for p in "$KV_META_PORT" "$KV_STORE_PORT" "$KV_METRICS_PORT"; do
        ss -ltn 2>/dev/null | grep -q ":$p " || miss="$miss $p"
      done
      if [ -n "$miss" ]; then
        echo "FATAL: 复用的 meta 容器 $KV_META_NAME 上这些端口没在听:$miss" >&2
        echo "       它启动时用的端口可能与本次不同。二选一：" >&2
        echo "       ① 改 KV_META_NAME 另起一个；② docker rm -f $KV_META_NAME 后重跑" >&2
        exit 6
      fi
      echo "[kv] meta 容器 $KV_META_NAME 在运行，端口 $KV_META_PORT/$KV_STORE_PORT/$KV_METRICS_PORT 均在听，复用"
    else
      want_ports=$(printf '%s\n' "$KV_META_PORT" "$KV_STORE_PORT" "$KV_METRICS_PORT" | sort -u | paste -sd, -)
      if [ "$meta_ports" != "$want_ports" ]; then
        echo "FATAL: 已存在的 meta 容器 $KV_META_NAME 端口是 [$meta_ports]，本次要 [$want_ports]" >&2
        echo "       复用它会导致引擎连不上。二选一：" >&2
        echo "       ① 改 KV_META_NAME 另起一个；② docker rm -f $KV_META_NAME 后重跑" >&2
        exit 6
      fi
      echo "[kv] meta 容器 $KV_META_NAME 已在运行且端口匹配 [$meta_ports]，复用（池内容保留）"
    fi
    return 0
  fi

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[kv] DRY_RUN：将启动 meta 容器 $KV_META_NAME（$KV_META_PORT/$KV_STORE_PORT/$KV_METRICS_PORT）"
    return 0
  fi

  # 新起一个之前，三个端口必须真的空闲：被别的东西占着的话，meta 会起在别人的
  # 端口上（或者起不来），而引擎要到 8 分钟后才报连不上。
  miss=""
  for p in "$KV_META_PORT" "$KV_STORE_PORT" "$KV_METRICS_PORT"; do
    ss -ltn 2>/dev/null | grep -q ":$p " && miss="$miss $p"
  done
  if [ -n "$miss" ]; then
    echo "FATAL: 要起 meta 容器，但这些端口已被占用:$miss" >&2
    echo "       查：ss -ltnp | grep -E ':(5100|6100|8100) '" >&2
    echo "       换端口：KV_META_PORT=... KV_STORE_PORT=... KV_METRICS_PORT=... 或 docker rm -f $KV_META_NAME" >&2
    exit 6
  fi

  docker rm -f "$KV_META_NAME" >/dev/null 2>&1 || true
  echo "[kv] 启动 meta 容器 $KV_META_NAME"
  # --restart unless-stopped：meta 必须比引擎活得久，否则引擎侧登记全丢
  docker run -d --name "$KV_META_NAME" --network host --restart unless-stopped \
    -e MMC_META_CONFIG_PATH="$KV_CONF_IN_CT/mmc-meta.conf" \
    -v "$KV_CONF_DIR:$KV_CONF_IN_CT:ro" \
    --entrypoint bash "$IMAGE" -c \
    'exec python -c "from memcache_hybrid import MetaService; MetaService.main()"' >/dev/null

  # 等 metrics 端口。起不来就别浪费 9 分钟去起 vllm。
  local ok=0 _
  for _ in $(seq 1 60); do
    if curl -sf --max-time 2 "http://127.0.0.1:${KV_METRICS_PORT}/metrics" >/dev/null 2>&1; then ok=1; break; fi
    sleep 1
  done
  if [ "$ok" != "1" ]; then
    echo "FATAL: meta 容器 60s 内未就绪（metrics :$KV_METRICS_PORT 不通）" >&2
    echo "       看：docker logs --tail 40 $KV_META_NAME" >&2
    exit 6
  fi
  echo "[kv] meta 就绪 :$KV_META_PORT/:$KV_STORE_PORT/:$KV_METRICS_PORT"
}

kv_offload_setup() {
  kv_offload_plan
  kv_offload_ensure_meta
}
