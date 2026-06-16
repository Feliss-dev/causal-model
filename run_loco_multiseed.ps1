# run_loco_multiseed.ps1
# Chay CausalHeteroGNN + Baseline GNN + MLP tren 3 LOCO fold x seed {1,2}
# (seed=42 da co san cho ca 3 mo hinh)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"   # bo qua counterfactual de nhanh hon

$folds = @(
    @{ tag="locoa"; dir="data/processed_loco_a" },
    @{ tag="locob"; dir="data/processed_loco_b" },
    @{ tag="lococ"; dir="data/processed_loco_c" }
)
$seeds = @(1, 2)

$total = $folds.Count * $seeds.Count * 2   # 2 scripts
$done  = 0
$start_all = Get-Date

foreach ($fold in $folds) {
    foreach ($seed in $seeds) {
        $run_tag = "_$($fold.tag)_s$seed"
        $env:GNN_INPUT_DIR        = $fold.dir
        $env:GNN_SEED             = "$seed"
        $env:GNN_RUN_TAG          = $run_tag
        $env:GNN_OOD_TRANSDUCTIVE = "0"

        # ── CausalHeteroGNN (06_train_gnn.py) ──────────────────────────────
        $out_causal = "results/metrics$run_tag.json"
        if (Test-Path $out_causal) {
            Write-Host "[SKIP] $out_causal already exists" -ForegroundColor Yellow
        } else {
            Write-Host "`n=== CausalHeteroGNN | $($fold.tag) | seed=$seed ===" -ForegroundColor Cyan
            $t0 = Get-Date
            uv run python pipeline/06_train_gnn.py
            $elapsed = [int]((Get-Date) - $t0).TotalSeconds
            Write-Host "[OK] $out_causal  (${elapsed}s)" -ForegroundColor Green
        }
        $done++
        Write-Host "Progress: $done / $total"

        # ── Baseline GNN + MLP (09_baselines_erm_mlp.py) ───────────────────
        $out_base = "results/baselines_erm_mlp$run_tag.json"
        if (Test-Path $out_base) {
            Write-Host "[SKIP] $out_base already exists" -ForegroundColor Yellow
        } else {
            Write-Host "`n=== Baseline GNN + MLP | $($fold.tag) | seed=$seed ===" -ForegroundColor Cyan
            $t0 = Get-Date
            uv run python pipeline/09_baselines_erm_mlp.py
            $elapsed = [int]((Get-Date) - $t0).TotalSeconds
            Write-Host "[OK] $out_base  (${elapsed}s)" -ForegroundColor Green
        }
        $done++
        Write-Host "Progress: $done / $total"
    }
}

$total_elapsed = [int]((Get-Date) - $start_all).TotalSeconds
Write-Host "`n=== ALL DONE in ${total_elapsed}s ===" -ForegroundColor Magenta
Write-Host "Ket qua moi:"
Get-ChildItem results/ | Where-Object { $_.Name -match "loco[abc]_s[12]" } | Sort-Object Name | Select-Object -ExpandProperty Name
