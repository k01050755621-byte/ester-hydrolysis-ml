#!/bin/bash
# DFT 校准监督器：worker 死掉自动重启（脚本在物种级有缓存，重启零损失）
#
# 背景：2026-09-28 21:02/21:09 两个 worker 被系统杀掉（疑似内存压力 / 背景任务被回收）。
# run_calib_expand.py 的 dft_opt.log（几何+SP）、pcm.log（PCM 单点）、xtbopt.xyz 都有缓存，
# 重启后已完成物种直接 [cache] 返回，只重算被杀时刻那一个物种。
#
# 用法（须以受管后台任务方式启动，nohup& 会被带走）：
#   bash ~/polywork/fix_20260928/calib_expand/supervise_dft.sh

cd "$HOME/polywork" || exit 1
PY="$HOME/miniconda3/envs/dft/bin/python"
DIR="$HOME/polywork/fix_20260928/calib_expand"
SCRIPT="$DIR/run_calib_expand.py"
SUP_LOG="$DIR/supervisor.log"

W1_IDS="C642 C7712 C8967 C12225 C1074 C1223 C2606 C6311 C5432"
W2_IDS="C672 C11896 C10302 C5327 C6488 C8188 C6533 C9132 C4141 C3482"

# 降低单 worker 资源占用（原 5 GB x 3 线程，疑似触发内存压力被杀）
export PSI4_MEM="3.5 GB"
export PSI4_THREADS="2"
export OMP_NUM_THREADS="2"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$SUP_LOG"; }

# PID 记录文件（用 pgrep -f 匹配易误伤自身命令行）
PIDW1="$DIR/.pid_w1"; PIDW2="$DIR/.pid_w2"

alive() {  # $1 = pidfile
  [ -f "$1" ] || return 1
  p=$(cat "$1")
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}

start() {  # $1 = worker 名, $2 = ID 串, $3 = pidfile
  WORKER="$1" $PY "$SCRIPT" $2 >> "$DIR/log_$1.log" 2>> "$SUP_LOG" &
  echo $! > "$3"
  log "启动 worker $1 (pid $(cat "$3"))，目标 $(echo $2 | wc -w | tr -d ' ') 个"
}

log "==== 监督器启动 ===="
start w1 "$W1_IDS" "$PIDW1"
start w2 "$W2_IDS" "$PIDW2"

while true; do
  sleep 60
  case "$(cat "$DIR/.stop" 2>/dev/null)" in 1) log "收到停止信号，退出"; exit 0;; esac
  if ! alive "$PIDW1"; then
    log "worker w1 已退出 → 重启（已完成物种走缓存）"
    start w1 "$W1_IDS" "$PIDW1"
  fi
  if ! alive "$PIDW2"; then
    log "worker w2 已退出 → 重启（已完成物种走缓存）"
    start w2 "$W2_IDS" "$PIDW2"
  fi
done
