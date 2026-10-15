#!/usr/bin/env bash
# Chạy toàn bộ lưới thí nghiệm UPFD (propagation/upfd_jobs.txt), 3 job song song.
# Từ project root:  bash propagation/run_upfd.sh
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/Scripts/python.exe
mkdir -p results/upfd/logs
rm -f results/upfd/logs/.done
# Giữ máy không ngủ trong lúc chạy (SetThreadExecutionState, tự hết hiệu lực khi process dừng)
$PY -c "
import ctypes, os, time
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
while not os.path.exists('results/upfd/logs/.done'): time.sleep(30)
" &
KEEP=$!
grep -v '^\s*$' propagation/upfd_jobs.txt | xargs -P 3 -I{} bash -c '
  name=$(echo "{}" | tr " " "_" | tr -d "-")
  $0 -u propagation/02_train_upfd.py {} > "results/upfd/logs/$name.log" 2>&1
  echo "done: {}"
' $PY
touch results/upfd/logs/.done
wait $KEEP
$PY propagation/03_upfd_report.py
echo ALL DONE
