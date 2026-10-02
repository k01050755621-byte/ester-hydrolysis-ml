#!/bin/bash
# DFT 校准扩容 · 启动/停止/查状态（幂等，可反复运行）
#
# 为什么需要它在**你自己的 Terminal** 里跑：
#   沙箱里以"受管后台任务"起的 DFT worker 会在约 4.5 小时后被回收
#   （2026-09-28 实测：w1 于 21:02、w2 于 21:09 被杀；relabel/接力器 3 小时内正常完成）。
#   launchd 在该环境被禁用（bootstrap/load 均报 EIO 5）。
#   你 Terminal 里 nohup 起的进程不在会话进程组内，不会被回收。
#
# 脚本在物种级有缓存（dft_opt.log / pcm.log / xtbopt.xyz），
# 任何时刻重启只损失"被杀时刻正在算的那一个物种"，已完成的不重算。
#
# 用法：
#   bash ~/polywork/fix_20260928/calib_expand/start_dft.sh          # 启动（已在跑则跳过）
#   bash ~/polywork/fix_20260928/calib_expand/start_dft.sh status   # 看进度
#   bash ~/polywork/fix_20260928/calib_expand/start_dft.sh stop     # 停止
set -u
DIR="$HOME/polywork/fix_20260928/calib_expand"
PY="$HOME/miniconda3/envs/dft/bin/python"
SCRIPT="$DIR/run_calib_expand.py"

W1_IDS="C642 C7712 C8967 C12225 C1074 C1223 C2606 C6311 C5432"
W2_IDS="C672 C11896 C10302 C5327 C6488 C8188 C6533 C9132 C4141 C3482"

nrun() { pgrep -f "run_calib_expand\.py C" 2>/dev/null | wc -l | tr -d ' '; }
ndone() { find "$DIR" -name dft_opt.log 2>/dev/null | wc -l | tr -d ' '; }
npcm()  { find "$DIR" -name pcm.log 2>/dev/null | wc -l | tr -d ' '; }

case "${1:-start}" in
  stop)
    pgrep -f "run_calib_expand\.py C" 2>/dev/null | while read -r p; do kill -TERM "$p" 2>/dev/null; done
    sleep 5
    echo "已发送停止信号，剩余进程：$(nrun)"
    exit 0 ;;
  status)
    echo "worker 进程数          : $(nrun)"
    echo "已完成几何+SP 物种数   : $(ndone)   (目标 57)"
    echo "已完成 PCM 单点物种数  : $(npcm)"
    echo "--- w1 日志尾 ---"; tail -4 "$DIR/log_w1.log" 2>/dev/null
    echo "--- w2 日志尾 ---"; tail -4 "$DIR/log_w2.log" 2>/dev/null
    exit 0 ;;
esac

if [ "$(nrun)" != "0" ]; then
  echo "已有 worker 在运行，跳过启动："
  pgrep -fl "run_calib_expand\.py C"
  exit 0
fi

# ★ 2026-09-28 21:50 修：psi4 的 scratch 目录（psi.XXXX）与 PCMSolver 的 PEDRA.OUT
#   都写在**进程 CWD**里，并且在收尾时要 unlink 它们。原先 CWD=~/polywork，
#   在沙箱内这些 unlink 会被拦（file-write-unlink denied）→ Fortran "File cannot
#   be deleted" → 整个 worker 崩。改到系统临时目录当 scratch，计算产物仍写回 $DIR
#   （脚本里所有输出路径都是绝对路径，不受 CWD 影响）。
SCRATCH="${TMPDIR:-/tmp}/polywork_dft_scratch"
mkdir -p "$SCRATCH"
cd "$SCRATCH" || exit 1
export PSI4_MEM="3.5 GB" PSI4_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

( WORKER=w1 nohup "$PY" "$SCRIPT" $W1_IDS >> "$DIR/launch_w1.out" 2>> "$DIR/launch_w1.err" & )
( WORKER=w2 nohup "$PY" "$SCRIPT" $W2_IDS >> "$DIR/launch_w2.out" 2>> "$DIR/launch_w2.err" & )
disown -a 2>/dev/null
sleep 10
echo "已启动 worker："
pgrep -fl "run_calib_expand\.py C"
echo
echo "查看进度: bash $DIR/start_dft.sh status"
echo "停止    : bash $DIR/start_dft.sh stop"
