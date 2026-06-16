# PIPELINE — Tái lập toàn bộ kết quả CausalHeteroGNN

Chạy **từ thư mục gốc dự án** (nơi chứa `data/`, `results/`, `models/`, `files/`).

```
Fakeddit/
├── data/               ← dữ liệu xử lý (do pipeline tạo ra)
├── results/            ← JSON kết quả + bảng cuối
├── models/             ← checkpoint .pt
├── figures/            ← ảnh PNG output (do pipeline tạo ra)
├── pipeline/           ← 12 scripts + run_all.ps1  (bạn đang ở đây)
└── files/              ← 5 scripts vẽ hình (gọi bởi script 11)
```

---

## Yêu cầu hệ thống

| Thành phần | Phiên bản tối thiểu | Ghi chú |
|---|---|---|
| Python | 3.10+ | quản lý bằng `uv` |
| PyTorch | 2.1+ | CPU hoặc CUDA |
| PyTorch-Geometric | 2.4+ | `torch_geometric`, `torch_scatter`, `torch_sparse` |
| sentence-transformers | 2.2+ | mã hóa tiêu đề bài (all-mpnet-base-v2) |
| transformers + CLIP | 4.35+ | `openai/clip-vit-base-patch32` |
| pandas, numpy, scikit-learn | mới nhất ổn định | |
| matplotlib | 3.8+ | vẽ hình trong `files/` |
| streamlit | 1.32+ | chỉ cần cho script 12 (dashboard) |
| Neo4j + GDS | 5.x | tùy chọn — fallback NetworkX nếu không có |

**Cài đặt nhanh:**
```powershell
uv sync          # đọc pyproject.toml / requirements.txt
# hoặc
pip install torch torch-geometric sentence-transformers transformers matplotlib streamlit
```

---

## Chạy nhanh — toàn bộ pipeline

```powershell
$env:PYTHONUTF8 = "1"      # bắt buộc trên Windows

# Chạy từng phase riêng (khuyến nghị lần đầu)
powershell -File pipeline/run_all.ps1 -Phase data
powershell -File pipeline/run_all.ps1 -Phase main
powershell -File pipeline/run_all.ps1 -Phase aggregate

# Hoặc chạy tất cả liên tiếp
powershell -File pipeline/run_all.ps1 -Phase all
```

Kết thúc bạn sẽ có:
- `results/final_tables.md` — bảng kết quả paper (mean±std 3 seeds)
- `figures/image1.png … image6.png` — 5 hình cho bài báo

---

## Chi tiết từng phase

### PHASE A — Dữ liệu (`-Phase data`)

> **Chạy 1 lần duy nhất.** Cần file TSV gốc của Fakeddit ở thư mục gốc.

#### Yêu cầu trước khi chạy

- `multimodal_train.tsv` và `multimodal_validate.tsv` ở thư mục gốc
  (tải từ [Fakeddit GitHub](https://github.com/entitize/Fakeddit))
- `multimodal_test_public.tsv` (tùy chọn — nếu không có sẽ dùng validate.tsv)
- Kết nối Internet để tải ảnh và model CLIP/mpnet

---

#### Script 01 — `01_prepare_data.py`

**Làm gì:**
- Tải ảnh từ URL (10 luồng song song), lọc bài có ảnh hợp lệ
- Mã hóa tiêu đề bằng `all-mpnet-base-v2` (768 chiều)
- Trích CLIP image embedding (512 chiều, L2 chuẩn hóa)
- Tính feature từ **chỉ tập train** (tránh data leakage):
  - User: `post_count`, `avg_score`, `avg_upvote_ratio`, `fake_rate`
  - Subreddit: `post_count`, `fake_ratio_real`, `avg_score`
  - Domain: `post_count`, `fake_ratio_real`, `avg_upvote_ratio`
- Chia split cân bằng: Train 5000 / Val 400 / Test 200 / OOD 600
  - OOD subreddits: `neutralnews` (Real) và `theonion` (Fake)
- Sinh CSV node/edge cho Neo4j và cho GNN

**Output:**

| File | Nội dung |
|---|---|
| `data/processed/sampled_master.csv` | Toàn bộ bài + nhãn split |
| `data/processed/posts.csv` | Node Post với features |
| `data/processed/users.csv` | Node User |
| `data/processed/subreddits.csv` | Node Subreddit |
| `data/processed/domains.csv` | Node Domain |
| `data/processed/images.csv` | Node Image |
| `data/processed/posted_by.csv` | Edge Post→User |
| `data/processed/posted_in.csv` | Edge Post→Subreddit |
| `data/processed/links_to.csv` | Edge Post→Domain |
| `data/processed/has_image.csv` | Edge Post→Image |
| `data/processed/member_of.csv` | Edge User→Subreddit |
| `data/processed/post_embeddings.npy` | Text embedding (N×768) |
| `data/processed/image_embeddings.npy` | CLIP embedding (N×512) |
| `data/images/` | Ảnh JPG đã tải |

---

#### Script 02 — `02_neo4j_import.py`

**Làm gì:**
- Kết nối Neo4j và import toàn bộ node/edge (batch 500)
- Chạy GDS: PageRank (Domain), Louvain community (Post), Betweenness (User), FastRP 64 chiều (Post)
- **Fallback NetworkX** nếu không có Neo4j: tính các chỉ số tương đương bằng NetworkX
- Ghi metadata Causal DAG vào Neo4j
- Merge kết quả GDS vào CSV enriched

**Neo4j (tùy chọn):**
```powershell
# Khởi động Neo4j qua Docker
docker compose up -d
# hoặc đặt biến môi trường nếu dùng instance khác
$env:NEO4J_URI      = "bolt://localhost:7887"
$env:NEO4J_USER     = "neo4j"
$env:NEO4J_PASSWORD = "password123"
```

**Output:**

| File | Nội dung |
|---|---|
| `data/processed/posts_enriched.csv` | Post + community_id (Louvain) |
| `data/processed/users_enriched.csv` | User + betweenness |
| `data/processed/domains_enriched.csv` | Domain + pagerank |
| `data/processed/subreddits_enriched.csv` | Subreddit enriched |
| `data/processed/images_enriched.csv` | Image enriched |
| `data/processed/post_fastrp.npy` | FastRP embedding (N×64) |

---

#### Script 03 — `03_clip_consistency.py`

**Làm gì:**
- Tính điểm nhất quán ngữ nghĩa tiêu đề–ảnh: `cosine(CLIP-text(title), CLIP-image)`
- Dùng làm đặc trưng phát hiện tin giả (tiêu đề không khớp ảnh → khả năng cao là fake)
- Copy sang `data/processed_confounded/` nếu thư mục đó tồn tại

**Output:**
- `data/processed/clip_cons.npy` — mảng float32 (N×1)

---

#### Script 04 — `04_make_confounded.py`

**Làm gì:**
- Tạo benchmark **Confounding-Shift** kiểu ColoredMNIST:
  - Thay subreddit thật bằng biến giả tổng hợp nhị phân (env ∈ {0, 1})
  - Train/Val/Test-seen: tương quan spurious ρ=0.9 (subreddit dự đoán label được)
  - OOD-test: tương quan đảo ngược ρ=0.1 (subreddit misleading)
- Model bị overfit vào subreddit sẽ sụp đổ trên OOD test

**Output:** Toàn bộ `data/processed_confounded/` — cấu trúc giống `data/processed/` nhưng với 2 subreddit tổng hợp.

---

### PHASE B — Huấn luyện & Đánh giá (`-Phase main`)

> Chạy **3 seeds × 2 protocols** → tổng 24 lần gọi script (12 train + 12 eval).  
> Thời gian ước tính: ~2–4 giờ (CPU), ~30–60 phút (GPU).

---

#### Script 05 — `05_train_gnn.py`

**Làm gì:**
- Xây dựng `HeteroData` graph (5 node types, 5 edge types)
- Kiến trúc **CausalHeteroGNN**:
  - Nhánh baseline: HeteroGraphSAGE full-graph → embedding `h_b`
  - Nhánh causal: cô lập Subreddit (causal cut) → embedding `h_c`
  - Gradient Reversal Layer (GRL α=2.0): huấn luyện `h_c` để **không** dự đoán được subreddit
  - Multi-task loss: 2-way classification + 6-way + adversarial confounder + orthogonality penalty
- Early stopping trên val loss (patience=30, max 300 epoch)
- Lưu checkpoint tốt nhất

**Lệnh chạy thủ công (1 seed):**
```powershell
$env:GNN_SEED = "42"
$env:GNN_RUN_TAG = "_main_s42"
$env:GNN_SKIP_EXPLAIN = "1"         # bỏ counterfactual khi sweep
uv run python pipeline/05_train_gnn.py
```

**Output:**
- `models/causal_gnn{RUN_TAG}.pt` — checkpoint tốt nhất
- `results/training_history.json` — loss/F1 theo epoch
- `results/metrics{RUN_TAG}.json` — metrics sơ bộ (chưa có LFR nếu SKIP_EXPLAIN=1)

---

#### Script 06 — `06_evaluate.py`

**Làm gì:**
- Nạp checkpoint từ script 05
- Calibrate temperature trên val set
- Đánh giá **hai chế độ**:
  - **Transductive**: full-graph message passing (accuracy cao hơn nhưng có thể inflated)
  - **Inductive (strict)**: che khuất edge của test Post → content-only, leak-free
- Tính OOD metrics: seen vs unseen subreddits
- Nếu `GNN_SKIP_EXPLAIN=0`: chạy counterfactual engine + tính LFR, ghi vào `metrics.json`

**Lệnh chạy thủ công:**
```powershell
# Cùng env-var với script 05
uv run python pipeline/06_evaluate.py

# Chạy lại seed 42 với LFR (cần cho fig4_robustness):
Remove-Item Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
$env:GNN_SEED = "42"; $env:GNN_RUN_TAG = "_bd_s42"
uv run python pipeline/06_evaluate.py
```

**Output:**
- `results/metrics{RUN_TAG}.json` — metrics đầy đủ gồm:
  ```
  {
    "baseline": { "seen": {...}, "unseen": {...}, "f1_drop_pct": ... },
    "causal":   { "seen": {...}, "unseen": {...}, "f1_drop_pct": ... },
    "lfr": {
      "baseline": { "lfr_subreddit": 0.301, "lfr_image": 0.04, "lfr_domain": 0.062 },
      "causal":   { "lfr_subreddit": 0.0,   "lfr_image": 0.08, "lfr_domain": 0.155 }
    }
  }
  ```
  *(trường `lfr` chỉ có khi `GNN_SKIP_EXPLAIN=0`)*
- `results/counterfactuals.json` — xác suất CF per-post
- `results/causal_paths.json` — attribution gradient per-post

---

#### Script 07 — `07_baselines_irm_eerm.py`

**Làm gì:**
- Huấn luyện **IRM** (Invariant Risk Minimization): penalty gradient orthogonality qua các environment (partition theo subreddit)
- Huấn luyện **EERM** (Explore-to-Extrapolate): K=3 môi trường ảo qua edge dropout, loss = mean + β×variance
- Dùng cùng backbone HeteroGraphSAGE như CausalHeteroGNN nhưng không có causal cut / GRL / multi-task

**Lệnh chạy thủ công:**
```powershell
$env:GNN_SEED = "42"; $env:GNN_RUN_TAG = "_stdood_s42"
uv run python pipeline/07_baselines_irm_eerm.py
```

**Output:**
- `results/baselines_irm_eerm{RUN_TAG}.json`
  ```
  { "irm": { "seen": {...}, "unseen": {...}, "f1_drop_pct": ... },
    "eerm": { ... } }
  ```

---

#### Script 08 — `08_baselines_erm_mlp.py`

**Làm gì:**
- Huấn luyện **ERM HeteroSAGE**: baseline GNN độc lập (không phải nhánh baseline bên trong CausalHeteroGNN)
- Huấn luyện **MLP content-only**: 3-layer MLP trên Post features, không dùng graph
- Cả hai dùng giao thức eval OOD giống nhau (inductive/transductive theo biến môi trường)

**Output:**
- `results/baselines_erm_mlp{RUN_TAG}.json`
  ```
  { "erm": { "seen": {...}, "unseen": {...}, "f1_drop_pct": ... },
    "mlp": { ... } }
  ```

---

#### Bảng RUN_TAG chuẩn (3 seeds, 2 protocols)

| Mục đích | `GNN_INPUT_DIR` | `GNN_OOD_TRANSDUCTIVE` | RUN_TAG | Script |
|---|---|---|---|---|
| CausalHeteroGNN — Held-Out | `data/processed` | `0` | `_main_s{42,1,2}` | 05→06 |
| ERM/MLP — Held-Out | `data/processed` | `0` | `_stdood_s{42,1,2}` | 08 |
| IRM/EERM — Held-Out | `data/processed` | `0` | `_stdood_s{42,1,2}` | 07 |
| CausalHeteroGNN — Conf-Shift | `data/processed_confounded` | `1` | `_bd_s{42,1,2}` | 05→06 |
| ERM/MLP — Conf-Shift | `data/processed_confounded` | `1` | `_conf_s{42,1,2}` | 08 |
| IRM/EERM — Conf-Shift | `data/processed_confounded` | `1` | `_conf_s{42,1,2}` | 07 |
| LFR (seed 42, cho fig4) | `data/processed_confounded` | `1` | `_bd_s42` | 06 (không skip-explain) |

---

### PHASE C — Tổng hợp & Hình (`-Phase aggregate`)

---

#### Script 09 — `09_worst_group.py`

**Làm gì:**
- Nạp tất cả checkpoint từ `models/` (không train lại)
- Tính **Worst-Group Accuracy** và **Avg-Group Accuracy** trên OOD test
- Group định nghĩa theo `(subreddit, label)`:
  - Standard OOD: 2 groups (neutralnews|Real, theonion|Fake)
  - Confounding-Shift: 4 groups (env×label)
- Tổng hợp mean±std qua 3 seeds

**Lệnh chạy:**
```powershell
# Standard OOD
uv run python pipeline/09_worst_group.py

# Confounding-Shift
$env:GNN_INPUT_DIR = "data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE = "1"
uv run python pipeline/09_worst_group.py
```

**Output:**
- `results/worst_group_stdood.json`
- `results/worst_group_conf.json`
  ```json
  {
    "summary": {
      "MLP":            { "worst_group_acc": [0.156, 0.031], "avg_group_acc": [0.565, 0.010] },
      "ERM":            { "worst_group_acc": [0.235, 0.079], "avg_group_acc": [0.732, 0.044] },
      "IRM":            { "worst_group_acc": [0.244, 0.071], "avg_group_acc": [0.743, 0.027] },
      "EERM":           { "worst_group_acc": [0.286, 0.013], "avg_group_acc": [0.770, 0.010] },
      "CausalHeteroGNN":{ "worst_group_acc": [0.378, 0.113], "avg_group_acc": [0.708, 0.045] }
    }
  }
  ```
  *(giá trị 0–1, nhân 100 để ra %)*

---

#### Script 10 — `10_final_tables.py`

**Làm gì:**
- Glob tất cả file `results/metrics_main_s*.json`, `baselines_*.json`, v.v.
- Tính mean±std qua các seed
- Xuất bảng Markdown + JSON nguồn của mọi số liệu trong paper

**Lệnh chạy:**
```powershell
uv run python pipeline/10_final_tables.py
```

**Output:**
- `results/final_tables.md` — **nguồn số liệu paper** (in ra console, lưu vào file)
- `results/final_tables.json`
  ```json
  {
    "standard_ood": [
      { "model": "MLP (content-only)", "seen_acc": [0.817, 0.006],
        "ood_acc": [0.605, 0.011], "ood_f1": [0.57, 0.01],
        "ood_auc": [0.685, 0.02], "f1_drop": [30.0, 1.5] },
      { "model": "ERM HeteroSAGE",  ... },
      { "model": "IRM",             ... },
      { "model": "EERM",            ... },
      { "model": "BaselineBranch*", ... },
      { "model": "CausalHeteroGNN", ... }
    ],
    "confounding_shift": [ ... ]
  }
  ```
  *(accuracy/f1/auc dạng 0–1; f1_drop đã nhân 100 = %)*

---

#### Script 11 — `11_generate_figures.py`

**Làm gì:**
- Gọi tuần tự 5 script trong thư mục `files/`:
  1. `fig1_neo4j_schema.py` — sơ đồ kiến trúc HIN (hardcoded, không cần data)
  2. `fig2_architecture.py` — sơ đồ CausalHeteroGNN (hardcoded)
  3. `fig3_ood_performance.py` — đọc `results/final_tables.json`
  4. `fig4_robustness.py` — đọc `results/worst_group_conf.json` + `results/metrics_bd_s42.json`
  5. `fig5_dashboard.py` — đọc cả 3 JSON trên
- Kiểm tra tồn tại tất cả output

**Yêu cầu trước khi chạy:**

| File | Sinh bởi |
|---|---|
| `results/final_tables.json` | Script 10 |
| `results/worst_group_conf.json` | Script 09 (conf protocol) |
| `results/metrics_bd_s42.json` có trường `lfr` | Script 06 (seed 42, không skip-explain) |

**Lệnh chạy:**
```powershell
uv run python pipeline/11_generate_figures.py
```

**Output:**

| File | Hình trong bài | Nội dung |
|---|---|---|
| `figures/image1.png` | Hình 1 | Sơ đồ property-graph HIN (Neo4j schema) |
| `figures/image2.png` | Hình 2 | Kiến trúc CausalHeteroGNN |
| `figures/image3.png` | Hình 3 | Hiệu năng 2 panel: Acc 3-nhóm + AUC/F1-drop |
| `figures/image5.png` | Hình 4 | Worst-Group Acc + Label-Flip Rate |
| `figures/image6.png` | Hình 5 | BI Dashboard 6 panel |

---

### PHASE D — Dashboard tương tác (`-Phase dashboard`, tùy chọn)

#### Script 12 — `12_dashboard.py`

**Làm gì:**
- Streamlit web app với 7 panel tương tác:
  1. Graph Overview (số node/edge, community)
  2. Model Performance (Accuracy, F1, AUC, confusion matrix)
  3. OOD Robustness (F1-drop seen vs unseen)
  4. Counterfactual Explorer
  5. Causal Path & Explanations
  6. Confounder Analysis
  7. BI Insights (top domain, user karma)
- Graceful degradation: hoạt động ngay cả khi không có Neo4j hay không có results JSON (dùng dummy data)

**Lệnh chạy:**
```powershell
uv run streamlit run pipeline/12_dashboard.py
# Mở trình duyệt tại http://localhost:8501
```

---

## Bảng biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GNN_SEED` | `42` | random seed; dùng `42`, `1`, `2` cho 3 seeds |
| `GNN_RUN_TAG` | `""` | hậu tố tên file output, ví dụ `_main_s42` |
| `GNN_INPUT_DIR` | `data/processed` | đổi sang `data/processed_confounded` cho Confounding-Shift |
| `GNN_OOD_TRANSDUCTIVE` | `0` | **đặt `1` khi dùng confounded** (confounder phải nhìn thấy) |
| `GNN_SKIP_EXPLAIN` | `0` | `1` = bỏ counterfactual/LFR → chạy nhanh khi sweep seed |
| `GNN_HIDDEN` | `96` | kích thước hidden layer |
| `GNN_DROPOUT` | `0.4` | dropout rate |
| `GNN_GRL_ALPHA` | `2.0` | cường độ Gradient Reversal Layer |
| `GNN_W_ADV` | `0.5` | trọng số loss adversarial confounder |
| `GNN_USE_FASTRP` | `0` | `1` chỉ dùng cho ablation rò rỉ FastRP (§5.5) |
| `GNN_NEUTRAL_DOMAIN` | `0` | `1` chỉ dùng cho ablation lịch sử nguồn (§5.6) |

⚠️ **Xóa env-var cũ khi đổi protocol** để tránh kết quả sai:
```powershell
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_SKIP_EXPLAIN,
            Env:GNN_SEED, Env:GNN_RUN_TAG -ErrorAction SilentlyContinue
```

---

## Kiểm tra kết quả (Verification)

Sau khi chạy phase `aggregate`, kiểm tra:

```powershell
# 1. Xem bảng kết quả chính
Get-Content results/final_tables.md

# 2. Kiểm tra worst-group đã có
Test-Path results/worst_group_stdood.json   # True
Test-Path results/worst_group_conf.json     # True

# 3. Kiểm tra 5 hình đã sinh
Get-ChildItem figures/ | Select-Object Name, @{N='KB';E={[int]($_.Length/1024)}}
# Phải thấy: image1.png  image2.png  image3.png  image5.png  image6.png
```

**Số liệu tham chiếu** (3-seed mean, Confounding-Shift):

| Model | OOD Acc% | AUC | F1-drop% |
|---|---|---|---|
| MLP (content-only) | ~59.5 | ~0.685 | ~30.0 |
| ERM HeteroSAGE | ~52.1 | ~0.511 | ~45.1 |
| IRM | ~54.1 | ~0.524 | ~43.2 |
| EERM | ~59.6 | ~0.694 | ~38.6 |
| **CausalHeteroGNN** | **~74–80** | **~0.85–0.92** | **~5–13** |

---

## Cấu trúc file output tổng hợp

```
results/
├── metrics_main_s42.json        # CausalHeteroGNN, Held-Out, seed 42
├── metrics_main_s1.json         # seed 1
├── metrics_main_s2.json         # seed 2
├── metrics_bd_s42.json          # CausalHeteroGNN, Conf-Shift, seed 42 (+lfr)
├── metrics_bd_s1.json           # seed 1
├── metrics_bd_s2.json           # seed 2
├── baselines_irm_eerm_stdood_s42.json
├── baselines_irm_eerm_stdood_s1.json
├── baselines_irm_eerm_stdood_s2.json
├── baselines_irm_eerm_conf_s42.json
├── baselines_irm_eerm_conf_s1.json
├── baselines_irm_eerm_conf_s2.json
├── baselines_erm_mlp_stdood_s42.json
├── baselines_erm_mlp_stdood_s1.json
├── baselines_erm_mlp_stdood_s2.json
├── baselines_erm_mlp_conf_s42.json
├── baselines_erm_mlp_conf_s1.json
├── baselines_erm_mlp_conf_s2.json
├── worst_group_stdood.json      # từ script 09
├── worst_group_conf.json        # từ script 09
├── final_tables.md              # ← nguồn số liệu paper
├── final_tables.json            # ← nguồn data cho scripts vẽ hình
├── counterfactuals.json         # CF per-post (seed 42, conf-shift)
└── causal_paths.json            # gradient attribution per-post
```

---

## Tài liệu liên quan

- `RUNNING_GUIDE.md` — hướng dẫn chi tiết + troubleshooting
- `files/README.md` — hướng dẫn chạy riêng các script vẽ hình
