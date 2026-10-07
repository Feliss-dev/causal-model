# Pipeline Run Guide — CausalHeteroGNN on Fakeddit

Step-by-step instructions to run the full 12-script pipeline from a clean environment.
Run **all commands from the project root** (`d:\Causal_Code\causal-model\`), not from inside `pipeline/`.

---

## Prerequisites

| Item | Requirement | Check |
|------|-------------|-------|
| Python | 3.12 (via `uv`) | `uv run python --version` |
| uv | latest | `uv --version` |
| Neo4j Docker | 5.12.0 running | `docker ps` → `fakeddit-neo4j` |
| `multimodal_train.tsv` | in project root | `Test-Path multimodal_train.tsv` |
| `multimodal_validate.tsv` | in project root | `Test-Path multimodal_validate.tsv` |

**Install dependencies (first time only):**
```powershell
cd "D:\Causal_Code\causal-model"
uv sync
```

**Start Neo4j (if not already running):**
```powershell
docker compose up -d
# Browser UI : http://localhost:7874  (neo4j / password123)
# Bolt URI   : bolt://localhost:7887
```

**Verify everything before running:**
```powershell
uv run python -c "import torch; import torch_geometric; import neo4j; import sentence_transformers; print('deps OK')"
uv run python -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://localhost:7887',auth=('neo4j','password123')); d.verify_connectivity(); print('Neo4j OK'); d.close()"
```

---

## Quick Start — Run All Phases at Once

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase all
```

Or run each phase separately (recommended for first run):

```powershell
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase data       # Phase A: ~1–3 hours
powershell -File pipeline/run_all.ps1 -Phase main       # Phase B: ~1–6 hours
powershell -File pipeline/run_all.ps1 -Phase aggregate  # Phase C: <5 minutes
powershell -File pipeline/run_all.ps1 -Phase dashboard  # Phase D: optional
```

---

## Phase A — Data Preparation *(run once)*

**Scripts:** `01` → `02` → `03` → `04`
**Input:** `multimodal_train.tsv`, `multimodal_validate.tsv`
**Time:** 1–3 hours (dominated by image download + CLIP encoding on CPU)

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase data
```

### Manual step-by-step (equivalent to `-Phase data`)

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"

uv run python pipeline/01_prepare_data.py     # → data/processed/*.csv, post_embeddings.npy, image_embeddings.npy
uv run python pipeline/02_neo4j_import.py     # → data/processed/*_enriched.csv (needs 01 output)
uv run python pipeline/03_clip_consistency.py # → data/processed/clip_cons.npy (needs 01 output)
uv run python pipeline/04_make_confounded.py  # → data/processed_confounded/ (needs 02 + 03 output)
```

Run these **in order, one at a time** — each script depends on the previous one's output:
- `02` reads the node/edge CSVs that `01` exported to `data/processed/`
- `03` reads the CLIP image/text embeddings that `01` computed
- `04` reads the `*_enriched.csv` files from `02` and `clip_cons.npy` from `03`

`01_prepare_data.py` is the slow step (~1–3h: image download + CLIP/mpnet encoding on CPU) and should only be run **once** — re-running it re-downloads images and re-samples the dataset. `02`–`04` are fast (seconds to low minutes) and safe to re-run individually if one fails, as long as `01`'s output is already on disk.

### What each script does

**`01_prepare_data.py`** — Load, filter, split, embed
- Reads both TSV files; filters posts that have real images
- Holds out `r/neutralnews` (all Real) and `r/theonion` (all Fake) as OOD test
- Samples: train 2500 real + 2500 fake, val 400, seen-test 200, OOD ~600
- Downloads ~7000 images with 15 parallel threads → `data/images/`
- Computes text embeddings with `all-mpnet-base-v2` (768-d) → `post_embeddings.npy`
- Computes image embeddings with `CLIP ViT-B/32` (512-d) → `image_embeddings.npy`
- Exports 8 node/edge CSVs → `data/processed/`

**`02_neo4j_import.py`** — Build the Heterogeneous Information Network
- Clears Neo4j, creates constraints for 5 node types (Post, User, Subreddit, Domain, Image)
- Imports nodes + edges in batches of 500
- Runs GDS algorithms: PageRank (domain authority), Louvain (community), Betweenness (user influence), FastRP 64-d (for leakage ablation only)
- Exports `*_enriched.csv` files with GDS results merged in
- Falls back to NetworkX automatically if Neo4j is unreachable

**`03_clip_consistency.py`** — Title-image semantic consistency score
- Computes cosine similarity between CLIP text embedding of title and CLIP image embedding
- Low score → title and image are semantically mismatched (misinformation signal)
- Output: `data/processed/clip_cons.npy` (also copied to `processed_confounded/`)

**`04_make_confounded.py`** — Confounding-shift benchmark dataset
- Replaces real subreddits with a synthetic binary confounder: `spur_fakebias` / `spur_realbias`
- Train/val/seen-test: confounder correlates with label at ρ=0.9 (near-perfect shortcut)
- OOD test: correlation is reversed to ρ=0.1 (model relying on shortcut will flip)
- All content (text/image/scalar/domain/user) is unchanged — only subreddit is swapped
- Output: `data/processed_confounded/` (drop-in replacement via `GNN_INPUT_DIR`)

**Verify Phase A completed:**
```powershell
Test-Path data/processed/posts_enriched.csv       # True
Test-Path data/processed/post_embeddings.npy      # True
Test-Path data/processed/image_embeddings.npy     # True
Test-Path data/processed/clip_cons.npy            # True
Test-Path data/processed_confounded/posts_enriched.csv  # True
```

---

## Phase B — Training & Evaluation *(2 protocols × 3 seeds = 24 runs)*

**Scripts:** `05` → `06` → `07` → `08` (× 6 seed/protocol combos) + 1 LFR re-eval
**Time:** 1–6 hours on CPU (2–5 min per training run)

### Tóm tắt nhanh: các thông số cần nhớ

| Hạng mục | Giá trị mặc định |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.005` (`GNN_LR`) |
| Weight decay | `5e-4` (`GNN_WD`) |
| Batch size | Không có mini-batch, train full-graph/full-batch |
| Hidden size | `96` (`GNN_HIDDEN`) |
| Dropout | `0.4` (`GNN_DROPOUT`) |
| GRL alpha | `2.0` (`GNN_GRL_ALPHA`) |
| Scheduler | `ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)` |

**Classifier / head structure**
- Encoder chung: node feature projection riêng từng type → 2 lớp HeteroSAGE (SAGEConv, aggr=`sum`).
- `causal_head` và `spurious_head`: MLP `96 → 96 → 96` với ReLU + Dropout.
- `baseline_clf_2way`, `causal_clf_2way`: MLP `96 → 96 → 2`.
- `baseline_clf_6way`, `causal_clf_6way`: MLP `96 → 96 → 6`.
- `confounder_clf`: MLP `96 → 48 → num_subreddits`, dùng cho `L_spurious` và `L_adv` qua GRL.

**Công thức L_ortho**
\[
L_{ortho} = \mathrm{mean}\left(|\cos(h_c, h_s)|\right)
\]

**Tổng loss**
\[
L = L_{base,2w} + 0.5L_{base,6w} + L_{causal,2w} + 0.5L_{causal,6w} + 0.5L_{spurious} + 0.5L_{adv} + 0.2L_{ortho}
\]

### Hai protocol này thực chất là gì? (đọc trước khi quyết định có chạy hay không)

Cả hai protocol dùng **cùng một kiến trúc model** (CausalHeteroGNN) và cùng script train/eval — điểm khác nhau duy nhất là **cách chia dữ liệu train/test** và **loại "bẫy" (confounder)** mà model phải đối mặt lúc test. Mục đích là kiểm tra xem model có đang học đúng bản chất tin giả/tin thật hay chỉ đang "ăn gian" bằng cách bám vào một tín hiệu giả (shortcut).

**Protocol 1 — Standard OOD / Held-Out (tag `_main_s*`, `_stdood_s*`)**
- Dùng dữ liệu gốc (`data/processed/`), subreddit thật (r/news, r/worldnews, ...).
- Test set là 2 subreddit **chưa từng xuất hiện lúc train**: `r/neutralnews` (toàn bài Real) và `r/theonion` (toàn bài Fake, vì đây là trang tin giả trào phúng).
- "Inductive": lúc test, các cạnh (edges) nối tới bài test bị che đi — model chỉ được nhìn nội dung bài viết (text/ảnh), không được "nhìn trộm" qua đồ thị xem hàng xóm của nó là ai.
- Câu hỏi cần trả lời: model có tổng quát hoá được sang subreddit hoàn toàn mới không, hay nó chỉ học thuộc lòng đặc điểm của các subreddit đã thấy?

**Protocol 2 — Confounding-Shift (tag `_bd_s*`, `_conf_s*`)**
- Dùng dữ liệu đã bị "đánh tráo" (`data/processed_confounded/`, tạo bởi script `04`): subreddit thật bị thay bằng nhãn giả `spur_fakebias` / `spur_realbias`.
- Lúc train: nhãn giả này **tương quan gần như tuyệt đối** với label thật (ρ=0.9) — tức là chỉ cần nhìn subreddit-giả cũng đoán đúng label ~90% mà không cần hiểu nội dung. Đây chính là cái bẫy (confounder).
- Lúc test (OOD): tương quan bị **đảo ngược** (ρ=0.1) — nếu model đã "ăn gian" bằng cách bám vào subreddit-giả lúc train, nó sẽ đoán sai gần hết lúc test.
- "Transductive": lúc test vẫn giữ nguyên cạnh đồ thị (không che), vì mục đích là xem model có bị confounder "dắt mũi" qua đường truyền tin trên đồ thị hay không.
- Câu hỏi cần trả lời: nếu bỏ đi cái shortcut dễ dàng (subreddit), model có vẫn đoán đúng dựa trên nội dung thật sự không, hay sụp đổ hoàn toàn?

**Tại sao cần cả 2, không chỉ 1?**
- Protocol 1 kiểm tra khả năng tổng quát hoá tự nhiên (generalization) sang miền dữ liệu mới.
- Protocol 2 kiểm tra khả năng chống lại một shortcut được cài đặt có chủ đích (causal robustness) — đây là phần chứng minh cho luận điểm chính của model "CausalHeteroGNN": nhánh causal (cắt bỏ cạnh Subreddit, dùng Gradient Reversal Layer) phải giữ được độ chính xác ở Protocol 2, trong khi baseline (ERM/MLP không có cơ chế causal) sẽ sụp điểm mạnh khi confounder bị đảo ngược.
- Cả 2 protocol đều chạy trên 3 seed (42, 1, 2) để lấy mean±std, vì tập OOD chỉ ~300–600 bài nên kết quả dao động khá nhiều giữa các seed nếu chỉ chạy 1 lần.

**Không muốn chạy?** Không sao — phần này chỉ để hiểu ý nghĩa số liệu trong `results/final_tables.md` (cột protocol, cột model) khi đọc kết quả đã có sẵn, không bắt buộc phải tự chạy lại. Nếu chỉ cần tái tạo nhanh, xem mục "Chỉ chạy seed 42" phía dưới.

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase main
```

### Manual step-by-step (equivalent to `-Phase main`)

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"   # skip counterfactual during sweep (faster)

# Clear stale env vars
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_NEUTRAL_DOMAIN,
            Env:GNN_USE_FASTRP, Env:GNN_SEED, Env:GNN_RUN_TAG -ErrorAction SilentlyContinue

# ── PROTOCOL 1: Standard OOD / Held-Out (inductive, content-only edges masked) ─
foreach ($s in @("42", "1", "2")) {
    $env:GNN_SEED = $s

    $env:GNN_RUN_TAG = "_main_s$s"
    uv run python pipeline/05_train_gnn.py        # → models/causal_gnn_main_s{s}.pt
    uv run python pipeline/06_evaluate.py         # → results/metrics_main_s{s}.json

    $env:GNN_RUN_TAG = "_stdood_s$s"
    uv run python pipeline/07_baselines_irm_eerm.py  # → results/baselines_irm_eerm_stdood_s{s}.json
    uv run python pipeline/08_baselines_erm_mlp.py   # → results/baselines_erm_mlp_stdood_s{s}.json
}

# ── PROTOCOL 2: Confounding-Shift (transductive, reversed confounder at test) ──
$env:GNN_INPUT_DIR        = "data/processed_confounded"
$env:GNN_OOD_TRANSDUCTIVE = "1"

foreach ($s in @("42", "1", "2")) {
    $env:GNN_SEED = $s

    $env:GNN_RUN_TAG = "_bd_s$s"
    uv run python pipeline/05_train_gnn.py        # → models/causal_gnn_bd_s{s}.pt
    uv run python pipeline/06_evaluate.py         # → results/metrics_bd_s{s}.json

    $env:GNN_RUN_TAG = "_conf_s$s"
    uv run python pipeline/07_baselines_irm_eerm.py  # → results/baselines_irm_eerm_conf_s{s}.json
    uv run python pipeline/08_baselines_erm_mlp.py   # → results/baselines_erm_mlp_conf_s{s}.json
}

# ── LFR re-eval: seed 42, conf-shift, WITH counterfactual (paper Table 2) ──────
Remove-Item Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
$env:GNN_SEED    = "42"
$env:GNN_RUN_TAG = "_bd_s42"
uv run python pipeline/06_evaluate.py            # → metrics_bd_s42.json gets "lfr" field
```

### What each script does

### Chỉ chạy seed 42 (bỏ qua sweep 3 seeds)

`run_all.ps1 -Phase main` luôn chạy cả 3 seeds (`42, 1, 2`) vì `$SEEDS` bị hard-code trong script — không có flag để giới hạn seed. Muốn chỉ chạy seed 42 (nhanh hơn ~3 lần, phù hợp để test nhanh trước khi chạy full sweep), chạy tay từng lệnh sau, **không dùng `run_all.ps1`**:

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"
$env:GNN_SEED = "42"

# Clear stale env vars trước khi bắt đầu
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_NEUTRAL_DOMAIN,
            Env:GNN_USE_FASTRP, Env:GNN_RUN_TAG -ErrorAction SilentlyContinue

# ── PROTOCOL 1: Standard OOD / Held-Out ────────────────────────────
$env:GNN_RUN_TAG = "_main_s42"
uv run python pipeline/05_train_gnn.py        # → models/causal_gnn_main_s42.pt
uv run python pipeline/06_evaluate.py         # → results/metrics_main_s42.json

$env:GNN_RUN_TAG = "_stdood_s42"
uv run python pipeline/07_baselines_irm_eerm.py  # → results/baselines_irm_eerm_stdood_s42.json
uv run python pipeline/08_baselines_erm_mlp.py   # → results/baselines_erm_mlp_stdood_s42.json

# ── PROTOCOL 2: Confounding-Shift ──────────────────────────────────
$env:GNN_INPUT_DIR        = "data/processed_confounded"
$env:GNN_OOD_TRANSDUCTIVE = "1"

$env:GNN_RUN_TAG = "_bd_s42"
uv run python pipeline/05_train_gnn.py        # → models/causal_gnn_bd_s42.pt
uv run python pipeline/06_evaluate.py         # → results/metrics_bd_s42.json

$env:GNN_RUN_TAG = "_conf_s42"
uv run python pipeline/07_baselines_irm_eerm.py  # → results/baselines_irm_eerm_conf_s42.json
uv run python pipeline/08_baselines_erm_mlp.py   # → results/baselines_erm_mlp_conf_s42.json

# ── LFR re-eval: seed 42, conf-shift, WITH counterfactual ──────────
Remove-Item Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
$env:GNN_RUN_TAG = "_bd_s42"
uv run python pipeline/06_evaluate.py            # → metrics_bd_s42.json gets "lfr" field
```

Lưu ý khi chỉ chạy 1 seed:
- `09_worst_group.py` và `10_final_tables.py` (Phase C) vẫn chạy được bình thường, nhưng `final_tables.md` sẽ báo mean±std chỉ trên 1 giá trị (n=1) thay vì 3 seeds — không dùng để báo cáo số liệu cuối cùng trong paper, chỉ để kiểm tra nhanh pipeline chạy đúng.
- Không cần sửa `run_all.ps1` — nếu muốn seed khác 42, chỉ cần đổi `$env:GNN_SEED` và các hậu tố `_s42` → `_s<seed>` trong `GNN_RUN_TAG` tương ứng.
- Vẫn phải nhớ set `$env:PYTHONUTF8 = "1"` và dọn env vars cũ trước khi đổi protocol/seed.

**`05_train_gnn.py`** — Train CausalHeteroGNN
- Architecture: 5 node types → Linear(→96d) → 2× HeteroSAGE layers
- **Causal branch**: encodes `G_causal = G \ {all Subreddit edges}` (backdoor cut)
- **Baseline branch**: encodes full graph `G`
- Gradient Reversal Layer (α=2.0) on causal head → adversarially removes confounder signal
- Orthogonality constraint: `|cos(h_causal, h_baseline)|` penalized
- Loss: `L_base_2w + 0.5·L_base_6w + L_causal_2w + 0.5·L_causal_6w + 0.5·L_adv + 0.2·L_ortho`
- Early stopping on val loss (patience=30, max 300 epochs)
- Output: `models/causal_gnn{RUN_TAG}.pt` + `results/metrics{RUN_TAG}.json`

**`06_evaluate.py`** — Full evaluation of the saved checkpoint
- Standard OOD (`GNN_OOD_TRANSDUCTIVE=0`): masks all test edges → inductive, content-only
- Conf-shift (`GNN_OOD_TRANSDUCTIVE=1`): keeps message passing → transductive (confounder visible)
- Temperature calibration on val set (does not touch test)
- If `GNN_SKIP_EXPLAIN=0`: runs counterfactual engine `do(image=∅)`, `do(domain=credible)`,
  `do(subreddit=neutral)` and computes Label Flip Rate (LFR)
- Updates (overwrites) `results/metrics{RUN_TAG}.json` — this is the authoritative metric file

**`07_baselines_irm_eerm.py`** — IRM and EERM baselines
- Same HeteroSAGE backbone as CausalHeteroGNN — no structural cut, no GRL, no ortho
- **IRM** (IRMv1, λ=100): environments = subreddit partitions; penalises gradient variance
- **EERM** (K=3 virtual envs via edge perturbation ≤0.3): penalises risk variance across envs
- Output: `results/baselines_irm_eerm{RUN_TAG}.json`

**`08_baselines_erm_mlp.py`** — ERM and MLP baselines
- **ERM**: standalone HeteroSAGE trained with plain cross-entropy (no causal interventions)
  — this is the "Baseline GNN" in the paper
- **MLP**: 3-layer MLP on Post features only (mpnet 768d + scalars) — no graph at all
  — serves as the content-only floor for both protocols
- Output: `results/baselines_erm_mlp{RUN_TAG}.json`

**Verify Phase B completed:**
```powershell
(Get-ChildItem results -Filter "metrics_main_s*.json").Count   # 3
(Get-ChildItem results -Filter "metrics_bd_s*.json").Count     # 3
(Get-ChildItem results -Filter "baselines_*.json").Count       # 12
(Get-ChildItem models  -Filter "*.pt").Count                   # >= 12
```

---

## Phase C — Aggregate & Generate Figures *(<5 minutes)*

**Scripts:** `09` (×2) → `10` → `11`

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase aggregate
```

### Manual step-by-step

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_SEED,
            Env:GNN_RUN_TAG, Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue

# Worst-group accuracy — Standard OOD
uv run python pipeline/09_worst_group.py              # → results/worst_group_stdood.json

# Worst-group accuracy — Confounding-Shift
$env:GNN_INPUT_DIR        = "data/processed_confounded"
$env:GNN_OOD_TRANSDUCTIVE = "1"
uv run python pipeline/09_worst_group.py              # → results/worst_group_conf.json

# Aggregate all metrics into final paper tables (mean±std, 3 seeds)
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE -ErrorAction SilentlyContinue
uv run python pipeline/10_final_tables.py             # → results/final_tables.md + final_tables.json

# Generate all 5 paper figures
uv run python pipeline/11_generate_figures.py         # → figures/image{1,2,3,5,6}.png
```

### What each script does

**`09_worst_group.py`** — Worst-group accuracy from saved checkpoints (no re-training)
- Standard OOD groups: `(neutralnews, Real)` and `(theonion, Fake)`
- Conf-shift groups: 4 combinations of (synthetic env × label); worst = reversed-correlation group
- Loads all available model checkpoints (causal, ERM, MLP, IRM, EERM × 3 seeds)
- Output: `results/worst_group_stdood.json` and `results/worst_group_conf.json`

**`10_final_tables.py`** — Master results table
- Reads all `metrics_*.json` and `baselines_*.json` files
- Aggregates mean±std across 3 seeds for every model × every protocol
- Also includes domain-ablation and worst-group rows
- Output: `results/final_tables.md` (human-readable) + `results/final_tables.json` (for figures)
- **This is the authoritative source of all numbers in the paper.**

**`11_generate_figures.py`** — Calls all 5 figure scripts in `files/`
- `fig1_neo4j_schema.py`   → `figures/image1.png` — Heterogeneous graph schema diagram
- `fig2_architecture.py`   → `figures/image2.png` — CausalHeteroGNN architecture
- `fig3_ood_performance.py`→ `figures/image3.png` — 5-model OOD performance comparison
- `fig4_robustness.py`     → `figures/image5.png` — Worst-group accuracy + Label Flip Rate
- `fig5_dashboard.py`      → `figures/image6.png` — BI Dashboard summary (6-panel)

**Verify Phase C completed:**
```powershell
Test-Path results/final_tables.md    # True
Test-Path results/final_tables.json  # True
Get-ChildItem figures -Filter "*.png" | Select-Object Name
# image1.png  image2.png  image3.png  image5.png  image6.png
```

**View results:**
```powershell
Get-Content results/final_tables.md
```

---

## Phase D — Interactive Dashboard *(optional)*

```powershell
cd "D:\Causal_Code\causal-model"
$env:PYTHONUTF8 = "1"
uv run streamlit run pipeline/12_dashboard.py
# Open: http://localhost:8501
```

**`12_dashboard.py`** — 7-panel Streamlit web app
- Panel 1: overall metrics table
- Panel 2: OOD accuracy bar chart
- Panel 3: worst-group comparison
- Panel 4: calibration curves
- Panel 5: counterfactual analysis (LFR)
- Panel 6: Neo4j knowledge graph explorer (requires Neo4j running)
- Panel 7: causal path attribution

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONUTF8` | — | **Always set to `1` on Windows** — prevents Vietnamese text from crashing |
| `GNN_SEED` | `42` | Random seed. Use `42`, `1`, `2` for the 3-seed sweep |
| `GNN_RUN_TAG` | `""` | Output file suffix, e.g. `_main_s42`. Must be unique per run |
| `GNN_INPUT_DIR` | `data/processed` | Switch to `data/processed_confounded` for Protocol 2 |
| `GNN_OOD_TRANSDUCTIVE` | `0` | Set `1` for confounding-shift (transductive eval) |
| `GNN_SKIP_EXPLAIN` | `0` | Set `1` to skip counterfactual/LFR (faster sweeps) |
| `GNN_HIDDEN` | `96` | GNN hidden dimension |
| `GNN_DROPOUT` | `0.4` | Dropout rate |
| `GNN_GRL_ALPHA` | `2.0` | Gradient Reversal Layer strength |
| `GNN_W_ADV` | `0.5` | Adversarial confounder loss weight |
| `GNN_LR` | `0.005` | Learning rate |
| `GNN_WD` | `5e-4` | Weight decay |
| `GNN_USE_FASTRP` | `0` | Set `1` only for leakage ablation (§5.4) — leaks labels |
| `GNN_CAUSAL_CUT` | `1` | Set `0` to disable backdoor cut (ablation) |
| `GNN_NEUTRAL_DOMAIN` | `0` | Set `1` to zero-out domain fake_ratio (ablation §5.6) |
| `NEO4J_URI` | `bolt://localhost:7887` | Neo4j Bolt endpoint |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password123` | Neo4j password |

**Always clear env vars between protocols:**
```powershell
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_NEUTRAL_DOMAIN,
            Env:GNN_USE_FASTRP, Env:GNN_SEED, Env:GNN_RUN_TAG,
            Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
```

---

## Results File → Paper Section Mapping

| Paper location | Source file |
|---------------|-------------|
| Table 1 — all rows | `results/final_tables.md` |
| Table 1 — CausalHeteroGNN | `results/metrics_main_s{42,1,2}.json` (held-out) / `metrics_bd_s{42,1,2}.json` (conf-shift) |
| Table 1 — ERM, MLP | `results/baselines_erm_mlp_{stdood,conf}_s{42,1,2}.json` |
| Table 1 — IRM, EERM | `results/baselines_irm_eerm_{stdood,conf}_s{42,1,2}.json` |
| §5.3 — baseline branch diagnostic (36.4% / AUC 0.314) | `metrics_bd_s*.json` → key `baseline` |
| §5.4 — FastRP leakage (94% vs 61%) | `metrics_s*.json` (leaky) vs `metrics_nofrp_s*.json` |
| Table 2 — Label Flip Rate | `results/metrics_bd_s42.json` → key `lfr` |
| §5.6 — domain ablation (74.2% → 58.8%) | `results/metrics_nd_s*.json` |
| Worst-group | `results/worst_group_{stdood,conf}.json` |

---

## Expected Final Directory Structure

```
causal-model/
├── data/
│   ├── images/                    ~7000 JPGs (~2–5 GB)
│   ├── processed/                 8 CSVs + post_embeddings.npy + image_embeddings.npy + clip_cons.npy
│   └── processed_confounded/      mirror with synthetic subreddit confounder
├── models/
│   ├── causal_gnn_main_s42.pt    causal_gnn_main_s1.pt    causal_gnn_main_s2.pt
│   ├── causal_gnn_bd_s42.pt      causal_gnn_bd_s1.pt      causal_gnn_bd_s2.pt
│   ├── irm_*.pt  eerm_*.pt  erm_*.pt  mlp_*.pt            (×6 seed/protocol each)
├── results/
│   ├── metrics_main_s{42,1,2}.json
│   ├── metrics_bd_s{42,1,2}.json         (s42 has "lfr" field)
│   ├── baselines_irm_eerm_{stdood,conf}_s{42,1,2}.json
│   ├── baselines_erm_mlp_{stdood,conf}_s{42,1,2}.json
│   ├── worst_group_stdood.json
│   ├── worst_group_conf.json
│   ├── final_tables.md            ← PRIMARY paper metrics source
│   └── final_tables.json
└── figures/
    ├── image1.png   Neo4j schema
    ├── image2.png   CausalHeteroGNN architecture
    ├── image3.png   OOD performance comparison
    ├── image5.png   Worst-group accuracy + LFR
    └── image6.png   BI Dashboard summary
```

---

## Troubleshooting

| Symptom | Cause & Fix |
|---------|-------------|
| `UnicodeEncodeError: cp1252` | Missing `$env:PYTHONUTF8="1"` — always set before any script |
| `Python was not found` | Use `uv run python ...`, not `python` directly |
| Conf-shift OOD accuracy ~17% (too low) | Stale protocol env vars — run the `Remove-Item Env:...` block |
| `Model checkpoint not found` in `06_evaluate.py` | `05_train_gnn.py` must run first with the same `GNN_RUN_TAG` |
| Neo4j connection refused | `docker compose up -d`; wait 30s; or ignore — `02_neo4j_import.py` falls back to NetworkX |
| `01_prepare_data.py` stops with CLIP/SentenceTransformer error | Fail-fast by design (prevents silent random embeddings). Reinstall `transformers`, `sentence-transformers`, `pillow` then re-run |
| Val F1 = 1.0 from epoch 1 (ERM, transductive) | Expected — subreddit shortcut is near-perfect in transductive; early stop uses val **loss**, not F1 |
| OOD accuracy varies ±8 pts between seeds | Normal — OOD set is only ~300 posts. Always report mean±std of 3 seeds |
| `FileNotFoundError: results/final_tables.json` when generating figures | Phase B must complete before Phase C |
| Phase B takes too long | Add `$env:GNN_HIDDEN="64"` to shrink model; or reduce `TRAIN_PER_CLASS` in `01_prepare_data.py` |
