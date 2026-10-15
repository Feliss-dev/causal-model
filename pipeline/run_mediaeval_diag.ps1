# run_mediaeval_diag.ps1 — chạy lại toàn bộ chẩn đoán CausalHeteroGNN trên MediaEval 2015
# Chạy từ project root:  powershell -File pipeline/run_mediaeval_diag.ps1
# Yêu cầu: ảnh dev đã giải nén vào data/mediaeval2015/dev_images
#   & "C:\Program Files\WinRAR\UnRAR.exe" x -o+ <devset>\MediaEval2015_DevSet_Images.rar data\mediaeval2015\dev_images\
# và langdetect:  uv pip install langdetect

$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "2"
$py = ".venv\Scripts\python.exe"

# Bước 0: dựng đồ thị + báo cáo dữ liệu -> results/mediaeval_diag/step0_report.md
& $py pipeline/14_mediaeval_graph.py

# Bước 1: các lần chạy chẩn đoán (5 seed 0–4, 200 epoch)
$runs = @(
    @("--run_name", "M",    "--model", "mlp"),
    @("--run_name", "A",    "--model", "baseline"),
    @("--run_name", "full"),
    @("--run_name", "B",    "--grl_alpha", "0"),
    @("--run_name", "C1",   "--grl_schedule", "dann", "--grl_alpha", "1.0"),
    @("--run_name", "C2",   "--grl_schedule", "dann", "--grl_alpha", "2.0"),
    @("--run_name", "D",    "--no_image_nodes"),
    @("--run_name", "E",    "--class_weight")
)
New-Item -ItemType Directory -Force results/mediaeval_diag/logs | Out-Null
foreach ($r in $runs) {
    & $py pipeline/15_train_mediaeval.py @r *> "results/mediaeval_diag/logs/$($r[1]).log"
}
# F: thêm sau khi có kết quả B–E, ví dụ:
# & $py pipeline/15_train_mediaeval.py --run_name F --no_image_nodes --class_weight

# Tổng hợp -> results/mediaeval_diag/summary.md + curves_ABC.png
& $py pipeline/16_mediaeval_diag_report.py
