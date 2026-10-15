#!/usr/bin/env bash
# Bổ sung IRM / EERM / causal có điều kiện cho bảng so sánh (propagation/upfd_jobs_compare.txt).
# Tự chờ run_upfd.sh xong (results/upfd/logs/runner.log có "ALL DONE") rồi mới chạy.
# Từ project root:  bash propagation/run_upfd_compare.sh
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=.venv/Scripts/python.exe
until grep -q "ALL DONE" results/upfd/logs/runner.log 2>/dev/null; do sleep 30; done
rm -f results/upfd/logs/.done2
$PY -c "
import ctypes, os, time
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
while not os.path.exists('results/upfd/logs/.done2'): time.sleep(30)
" &
KEEP=$!
grep -v '^\s*$' propagation/upfd_jobs_compare.txt | xargs -P 3 -I{} bash -c '
  name=$(echo "{}" | tr " " "_" | tr -d "-")
  $0 -u propagation/02_train_upfd.py {} > "results/upfd/logs/$name.log" 2>&1
  echo "done: {}"
' $PY
touch results/upfd/logs/.done2
wait $KEEP
$PY propagation/03_upfd_report.py > /dev/null
$PY propagation/04_upfd_comparison.py
echo COMPARE DONE
