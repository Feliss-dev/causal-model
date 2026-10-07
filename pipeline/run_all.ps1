# run_all.ps1 — Chạy tuần tự pipeline tái lập kết quả.
# Dùng:  powershell -File pipeline/run_all.ps1 -Phase <data|main|aggregate|dashboard|all>
# LƯU Ý: chạy từ THƯ MỤC GỐC dự án. Phase "data" cần multimodal_*.tsv (+ Docker cho Neo4j).
#
# -SkipData : chỉ dùng với -Phase all — bỏ qua Phase A (01-04), chạy thẳng main+aggregate
#             trên data/processed/ đang có sẵn. Dùng khi đã có dữ liệu (gốc hoặc khôi phục
#             từ backup) và không muốn tải lại ảnh / resample lại dataset.
# -Force    : bỏ qua cảnh báo resample khi data/processed/posts_enriched.csv đã tồn tại.
#
# Cấu trúc pipeline (12 scripts):
#   01 prepare_data       06 evaluate
#   02 neo4j_import       07 baselines_irm_eerm
#   03 clip_consistency   08 baselines_erm_mlp
#   04 make_confounded    09 worst_group
#   05 train_gnn          10 final_tables
#                         11 generate_figures
#                         12 dashboard (Streamlit, optional)
param([string]$Phase = "aggregate", [switch]$SkipData, [switch]$Force)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$SEEDS = @("42", "1", "2")

function Clear-GnnEnv {
    Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_USE_CLIPCONS,
                Env:GNN_NEUTRAL_DOMAIN, Env:GNN_USE_FASTRP, Env:GNN_AUTOCUT,
                Env:GNN_GROUPDRO, Env:GNN_SKIP_EXPLAIN, Env:GNN_SEED, Env:GNN_RUN_TAG `
                -ErrorAction SilentlyContinue
}
function Run($script) {
    Write-Host ">>> uv run python pipeline/$script  [tag=$($env:GNN_RUN_TAG) seed=$($env:GNN_SEED)]" -ForegroundColor Cyan
    uv run python "pipeline/$script"
    if ($LASTEXITCODE -ne 0) { throw "FAILED: $script" }
}

switch ($Phase) {

  "data" {        # PHASE A — chạy 1 lần; cần file TSV + Docker/Neo4j
    if ((Test-Path "data/processed/posts_enriched.csv") -and -not $Force) {
      Write-Host "CẢNH BÁO: data/processed/posts_enriched.csv đã tồn tại." -ForegroundColor Yellow
      Write-Host "Chạy lại Phase A sẽ TẢI LẠI ẢNH và RESAMPLE LẠI dataset (không deterministic" -ForegroundColor Yellow
      Write-Host "do phụ thuộc mạng khi tải ảnh) — ghi đè dữ liệu hiện có, không thể khôi phục." -ForegroundColor Yellow
      Write-Host "Dùng -Force để xác nhận chạy lại, hoặc dùng -Phase all -SkipData nếu chỉ" -ForegroundColor Yellow
      Write-Host "muốn chạy lại Phase B+C trên dữ liệu đang có." -ForegroundColor Yellow
      throw "Huỷ Phase A để tránh ghi đè dữ liệu hiện có. Thêm -Force nếu chắc chắn muốn resample."
    }
    Clear-GnnEnv
    Run "01_prepare_data.py"
    Run "02_neo4j_import.py"
    Run "03_clip_consistency.py"
    Run "04_make_confounded.py"
  }

  "main" {        # PHASE B — bảng chính (2 protocol × 3 seeds)
    Clear-GnnEnv; $env:GNN_SKIP_EXPLAIN = "1"
    foreach ($s in $SEEDS) {           # Held-Out (inductive)
      $env:GNN_SEED = $s
      $env:GNN_RUN_TAG = "_main_s$s";   Run "05_train_gnn.py"; Run "06_evaluate.py"
      $env:GNN_RUN_TAG = "_stdood_s$s"; Run "07_baselines_irm_eerm.py"; Run "08_baselines_erm_mlp.py"
    }
    $env:GNN_INPUT_DIR = "data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE = "1"
    foreach ($s in $SEEDS) {           # Confounding-Shift (transductive)
      $env:GNN_SEED = $s
      $env:GNN_RUN_TAG = "_bd_s$s";   Run "05_train_gnn.py"; Run "06_evaluate.py"
      $env:GNN_RUN_TAG = "_conf_s$s"; Run "07_baselines_irm_eerm.py"; Run "08_baselines_erm_mlp.py"
    }
    # LFR (Panel b của fig4): chạy lại eval seed 42 KHÔNG skip-explain để có trường lfr
    Remove-Item Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
    $env:GNN_SEED = "42"; $env:GNN_RUN_TAG = "_bd_s42"
    Run "06_evaluate.py"
  }

  "aggregate" {   # PHASE C — worst-group, bảng cuối, sinh hình
    Clear-GnnEnv
    Run "09_worst_group.py"
    $env:GNN_INPUT_DIR = "data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE = "1"
    Run "09_worst_group.py"
    Clear-GnnEnv
    Run "10_final_tables.py"
    Run "11_generate_figures.py"
    Write-Host "`nXong. Xem results/final_tables.md và figures/" -ForegroundColor Green
  }

  "dashboard" {   # PHASE D — Streamlit interactive (optional)
    Clear-GnnEnv
    Write-Host "Khởi động Streamlit dashboard..." -ForegroundColor Cyan
    uv run streamlit run "pipeline/12_dashboard.py"
  }

  "all" {
    $phases = if ($SkipData) { @("main", "aggregate") } else { @("data", "main", "aggregate") }
    if ($SkipData) {
      Write-Host "-SkipData: bỏ qua Phase A, dùng data/processed/ hiện có." -ForegroundColor Cyan
    }
    foreach ($p in $phases) {
      & $PSCommandPath -Phase $p -Force:$Force
    }
    Write-Host "`nPipeline hoàn tất. Chạy -Phase dashboard để xem dashboard." -ForegroundColor Green
  }

  default {
    Write-Host "Phase không hợp lệ. Chọn: data | main | aggregate | dashboard | all" -ForegroundColor Yellow
    Write-Host "Cờ tuỳ chọn: -SkipData (chỉ với -Phase all, bỏ qua Phase A) | -Force (bỏ qua cảnh báo resample)" -ForegroundColor Yellow
  }
}
