#!/usr/bin/env bash
# =============================================================================
# base.sh —— 读 base.conf，并给出"基座相关"的三个派生量
#
#   source "<补丁仓库>/base.sh"
#
# 提供的函数/变量（都在 source 后可用）：
#   BASE_*                        base.conf 里的原值（解析后）
#   BASE_TREE                    = <本分支>/<BASE_TREE_DIR>（绝对路径，树产物目录）
#   base_pgo_dest <image>        探测该镜像里 libpython 的落点（PGO 挂载目标）
#
# ★ 为什么要有这一层：此前镜像名写在 4 处、python 版本号写在 3 处、树目录写在 3 处，
#   换基座时漏掉任何一处都是**静默失效**（最典型：PGO 挂到不存在的目录，
#   docker 自动建一个空目录、挂载"成功"、解释器根本不加载它 ⇒ 白丢 −3.2~5.1%，
#   日志里一个字都不报）。现在这些值只在 base.conf 出现一次。
#
# ★ 本文件必须能容忍"补丁仓库不在"：三个入口都可能单独被调用。
#   所以它只做"能算出就赋值"的事，绝不 exit；缺 base.conf 时全部留空，
#   由调用方用自己的兜底默认值（保持旧行为）。
# =============================================================================

BASE_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_CONF="$BASE_SH/base.conf"
BASE_SH_VERSION=1

: "${BASE_IMAGE:=}"
: "${BASE_IMAGE_ID:=}"
: "${BASE_UPSTREAM_COMMIT:=}"
: "${BASE_VLLM_COMMIT:=}"
: "${BASE_CANN:=}"
: "${BASE_PYTHON_VER:=}"
: "${BASE_TRITON_ASCEND:=}"
: "${BASE_TREE_DIR:=}"
: "${BASE_PGO_SO_REL:=}"
: "${BASE_PGO_DEST_HINT:=}"
: "${BASE_PGO_ARTIFACT_PYVER:=}"

if [ -f "$BASE_CONF" ]; then
  # 逐行解析而不是 `source`：base.conf 是我们自己的、只含赋值，但用 allowlist 解析
  # 可以保证它永远**不被执行**（将来有人往里写 $(...) 也不会变成命令注入）。
  #
  # ⚠ 引号处理必须严格分两种：
  #   带引号的值 —— 引号内**原样保留**，允许行内 `#`（实测踩过：早期版本把
  #                `BASE_PYTHON_VER='3.12.16'   # rc2 是 3.12.13` 整段收进变量，
  #                 于是版本比较永远成立/不成立，静默做出错误判断）；
  #   不带引号   —— 按 shell 惯例，`#` 起才是注释，先截掉。
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in ''|'#'*) continue ;; esac
    _k="${_line%%=*}"
    [ "$_k" = "$_line" ] && continue            # 没有 = 的行跳过
    _v="${_line#*=}"
    _k="${_k//[[:space:]]/}"
    _v="${_v#"${_v%%[![:space:]]*}"}"           # ltrim
    case "$_v" in
      \'*) # 单引号：取下一个 ' 之前的内容，原样（不再剥 #）
           _v="${_v#\'}"; _v="${_v%%\'*}" ;;
      \"*) # 双引号：同上
           _v="${_v#\"}"; _v="${_v%%\"*}" ;;
      *)   # 无引号：` #` 之后是注释
           case "$_v" in
             *[[:space:]]#*) _v="${_v%%[[:space:]]#*}" ;;
           esac
           _v="${_v%"${_v##*[![:space:]]}"}" ;;    # rtrim
    esac
    case "$_k" in
      BASE_IMAGE|BASE_IMAGE_ID|BASE_UPSTREAM_COMMIT|BASE_VLLM_COMMIT|BASE_CANN|\
      BASE_PYTHON_VER|BASE_TRITON_ASCEND|BASE_TREE_DIR|BASE_PGO_SO_REL|\
      BASE_PGO_DEST_HINT|BASE_PGO_ARTIFACT_PYVER)
        printf -v "$_k" '%s' "$_v" ;;
      *) : ;;                                   # 不认识的键静默忽略（向前兼容）
    esac
  done < "$BASE_CONF"
  unset _line _k _v
fi

# 树产物目录：相对补丁仓库根，按分支分家（tree/、tree-v0300/ …）
if [ -n "${BASE_TREE_DIR:-}" ]; then
  BASE_TREE="$BASE_SH/$BASE_TREE_DIR"
else
  BASE_TREE=""
fi

# PGO 产物绝对路径（相对补丁仓库根解析）。产物按 python 版本分家
# （optim/pgo/ = 3.12.13 自编，optim/pgo-3.12.16/ = 3.12.16 预编），
# 各基座互不覆盖 —— 否则切分支重建 PGO 会悄悄冲掉另一条线在用的产物。
if [ -n "${BASE_PGO_SO_REL:-}" ]; then
  BASE_PGO_SO="$BASE_SH/$BASE_PGO_SO_REL"
else
  BASE_PGO_SO=""
fi

# 本分支基座的 Python 版本（供 PGO 落点兜底）
BASE_PY_VER_EFF="${BASE_PYTHON_VER:-3.12.13}"

# ---------------------------------------------------------------------------
# base_pgo_dest <image>  —— 探测镜像里 libpython 的绝对落点
#
# 为什么必须探测：python 的次版本号会随基座变（3.12.13 → 3.12.16），而 PGO 是
# 把宿主编好的 libpython **挂到镜像里那个文件的位置**上覆盖。路径写死成旧版本时：
#   docker 会**自动创建**一个不存在的父目录，挂载照常"成功"，
#   但解释器加载的是镜像自带的库（DT_RPATH 指向真实版本目录）⇒ PGO 完全没生效，
#   而 `docker inspect`、启动日志、`sys.version` 都看不出异常（除非逐条 ldd）。
#
# 探测失败时退回到 BASE_PGO_DEST_HINT / 版本号拼出的路径，并**显式告警**——
# 宁可吵一声，也不要静默挂到空目录。
# ---------------------------------------------------------------------------
base_pgo_dest() {
  local img="${1:-${IMAGE:-}}"
  local dest=""

  # 显式覆盖优先（PGO_DEST 是旧有逃生阀，保持兼容）
  if [ -n "${PGO_DEST:-}" ]; then
    printf '%s' "$PGO_DEST"; return 0
  fi

  if [ -n "$img" ] && command -v docker >/dev/null 2>&1; then
    # 一次 0.7 s 的容器启动。相比 600 s 的起服可以忽略，而它换来的是
    # "换基座不用改脚本"。输出必须形如 /*.so* 才算数：DRY_RUN 会把 docker
    # 换成只回显命令的替身，那种情况下这里必须走到兜底分支。
    local dk=docker
    if [ "${DRY_RUN:-0}" = "1" ] && [ -x /usr/bin/docker ]; then dk=/usr/bin/docker; fi
    dest=$($dk run --rm --entrypoint python3 "$img" -c '
import os, sysconfig
d = sysconfig.get_config_var("LIBDIR")
n = sysconfig.get_config_var("INSTSONAME") or ("libpython%s.so.1.0" % sysconfig.get_config_var("VERSION"))
print(os.path.join(d, n) if d and n else "")' 2>/dev/null | tail -1 || true)
    case "$dest" in
      /*.so*) printf '%s' "$dest"; return 0 ;;
      *)      dest="" ;;
    esac
  fi

  if [ -n "${BASE_PGO_DEST_HINT:-}" ]; then
    # DRY_RUN 下 docker 是只回显命令的替身（见工作区根 start-server.sh），
    # 探测必然拿不到输出 —— 那是预期行为，不是故障，别刷警告吓人。
    [ "${DRY_RUN:-0}" = "1" ] || {
      echo "[base] 警告：探测不到 $img 里的 libpython 落点，退回 base.conf 的提示路径 $BASE_PGO_DEST_HINT" >&2
      echo "[base]       PGO 可能挂到不存在的目录（静默失效）。确认后用 PGO_DEST=... 覆盖。" >&2
    }
    printf '%s' "$BASE_PGO_DEST_HINT"; return 0
  fi

  dest="/usr/local/python${BASE_PY_VER_EFF}/lib/libpython${BASE_PY_VER_EFF%.*}.so.1.0"
  [ "${DRY_RUN:-0}" = "1" ] || {
    echo "[base] 警告：探测不到 $img 里的 libpython 落点，按 base.conf 版本号拼出 $dest" >&2
    echo "[base]       PGO 可能挂到不存在的目录（静默失效）。确认后用 PGO_DEST=... 覆盖。" >&2
  }
  printf '%s' "$dest"
}

# ---------------------------------------------------------------------------
# base_jemalloc <image> —— 探测镜像内可用的 jemalloc，返回容器内路径
#
# 为什么必须探测：LD_PRELOAD 指向不存在的文件时 ld.so 只打一行
#   "cannot be preloaded ... ignored" 就继续跑 —— **静默回退到 glibc malloc**。
# 每进程一行、8 rank 起服日志里几十行，但服务照常可用，很容易被当成噪声忽略。
# rc2 镜像里有 /usr/lib/aarch64-linux-gnu/libjemalloc.so.2，v0.30.0 镜像**没有**，
# 同一个写死路径于是从"生效"变成"静默失效"。
#
# 找不到任何可用的 jemalloc 时返回空串（调用方据此不传 LD_PRELOAD），
# 而不是传一个必然失败的路径。
# ---------------------------------------------------------------------------
base_jemalloc() {
  local img="${1:-${IMAGE:-}}"
  local p=""

  # 显式覆盖优先（JEMALLOC=... 是旧有逃生阀，保持兼容；空串 = 显式关闭）
  if [ -n "${JEMALLOC+x}" ]; then
    printf '%s' "${JEMALLOC}"
    return 0
  fi

  # DRY_RUN 下 PATH 里的 docker 是"只回显命令"的替身（见工作区根 start-server.sh），
  # 用它探测只会拿到一坨命令文本。这里显式走 /usr/bin/docker，让**预览与实跑一致**。
  local dk=docker
  if [ "${DRY_RUN:-0}" = "1" ] && [ -x /usr/bin/docker ]; then dk=/usr/bin/docker; fi

  if [ -n "$img" ] && command -v docker >/dev/null 2>&1; then
    # 一次 ~0.7 s 的容器启动；候选顺序 = 通用路径在前，镜像自带的后备在后。
    p=$($dk run --rm --entrypoint bash "$img" -c '
for f in /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
         /usr/lib/aarch64-linux-gnu/libjemalloc.so \
         /usr/lib64/libjemalloc.so.2 \
         /usr/local/Ascend/cann-*/aarch64-linux/lib64/libjemalloc.so; do
  [ -f "$f" ] && { printf "%s" "$f"; break; }
done' 2>/dev/null | tail -1 || true)
    case "$p" in
      /*) printf '%s' "$p"; return 0 ;;
      *)  p="" ;;
    esac
  fi

  [ "${DRY_RUN:-0}" = "1" ] || \
    echo "[base] 注意：$img 里找不到 jemalloc，本次不设 LD_PRELOAD（回退 glibc malloc）" >&2
  printf '%s' ""
}

# base_tree_dir —— 树产物目录名（相对补丁仓库根），供文档/脚本打印
base_tree_dir() {
  printf '%s' "${BASE_TREE_DIR:-tree}"
}
