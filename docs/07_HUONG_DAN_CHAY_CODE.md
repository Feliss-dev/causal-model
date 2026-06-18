# 07 — Hướng dẫn chạy code Pipeline từng bước

> Tài liệu này mô tả cách chạy **đúng thứ tự** 4 file pipeline chính và giải thích
> tại sao mỗi bước phải chạy trước bước kia. Tên file khớp với code trong repo hiện tại.

---

## Tổng quan thứ tự bắt buộc

```
[Bước 1]  pipeline/01_prepare_data.py
    ↓  tạo data/processed/*.csv và *.npy
[Bước 2]  pipeline/02_neo4j_import.py
    ↓  tạo data/processed/*_enriched.csv và post_fastrp.npy
[Bước 3]  pipeline/05_train_gnn.py
    ↓  tạo models/causal_gnn.pt và results/metrics.json
[Bước 4]  pipeline/06_evaluate.py
    ↓  tạo results/counterfactuals.json và causal_paths.json
```

**Không được bỏ qua bước nào** — mỗi bước đọc đầu ra của bước trước.

---

## Yêu cầu môi trường

### Phần mềm cần có

| Phần mềm | Phiên bản | Lý do cần |
|---------|-----------|-----------|
| Python | 3.12 | runtime chính |
| uv | mới nhất | quản lý môi trường ảo, cài packages |
| Docker Desktop | bất kỳ | chạy Neo4j (chỉ cần cho Bước 2) |
| RAM | ≥ 8 GB | CLIP model + PyTorch chạy trên CPU |
| Disk | ≥ 5 GB | ảnh tải về + embeddings .npy |

### Cài dependencies một lần duy nhất

Chạy từ thư mục gốc dự án (`Fakeddit/`):

```powershell
cd "D:\B_Learn_IT_on_Tube\Teacher_PA\DE_TAI_BAO_FAIR\Fakeddit"
uv sync
```

`uv sync` đọc `pyproject.toml` và `uv.lock`, tạo `.venv` với tất cả packages:
`torch`, `torch-geometric`, `sentence-transformers`, `transformers`, `Pillow`,
`neo4j`, `networkx`, `scikit-learn`, `pandas`, `numpy`, v.v.

### Biến môi trường bắt buộc (LUÔN set trước khi chạy)

```powershell
$env:PYTHONUTF8 = "1"
```

Bắt buộc trên Windows: script dùng tiếng Việt, console mặc định cp1252 sẽ crash nếu thiếu.

### File dữ liệu gốc (đặt vào thư mục gốc)

```
Fakeddit/
├── multimodal_train.tsv       ← ~155 MB, tải từ Fakeddit dataset
└── multimodal_validate.tsv    ← ~16 MB
```

---

## Bước 1 — Chuẩn bị dữ liệu

**File:** `pipeline/01_prepare_data.py`

### Bước này làm gì?

```
multimodal_train.tsv + multimodal_validate.tsv   (dữ liệu thô)
        │
        ▼ 1. Lọc bài CÓ ẢNH (hasImage=True, image_url hợp lệ)
        │
        ▼ 2. Tách OOD: neutralnews (Real) + theonion (Fake)
        │    → Hai subreddit này KHÔNG được dùng để train, chỉ cho OOD test
        │
        ▼ 3. Balanced sampling
        │    train:      2500 Real + 2500 Fake = 5000
        │    val:          200 Real +  200 Fake =  400
        │    seen-test:    100 Real +  100 Fake =  200
        │    OOD-test:  ~150 neutralnews + ~150 theonion = ~300
        │
        ▼ 4. Tải ảnh song song (15 luồng)
        │    → data/images/{post_id}.jpg
        │
        ▼ 5. Encode tiêu đề bằng SentenceTransformer all-mpnet-base-v2
        │    → post_embeddings.npy   shape (N, 768)
        │
        ▼ 6. Encode ảnh bằng CLIP ViT-B/32
        │    → image_embeddings.npy  shape (N, 512)
        │
        ▼ 7. Tính feature thống kê từ TRAIN ONLY (tránh data leakage)
        │    User: fake_rate, avg_score...
        │    Subreddit: fake_ratio_real...
        │    Domain: fake_ratio_real, avg_upvote_ratio...
        │
        ▼ 8. Xuất CSV nodes + edges
             data/processed/posts.csv, users.csv, subreddits.csv,
             domains.csv, images.csv,
             posted_by.csv, posted_in.csv, links_to.csv,
             has_image.csv, member_of.csv
```

### Lệnh chạy

```powershell
$env:PYTHONUTF8 = "1"
uv run python pipeline/01_prepare_data.py
```

### Thời gian ước tính

| Giai đoạn | Thời gian |
|-----------|-----------|
| Tải ảnh (~5000 ảnh, 15 luồng) | 15–40 phút |
| CLIP embedding (CPU, batch 32) | 10–20 phút |
| SentenceTransformer encoding | 5–10 phút |
| **Tổng** | **30–70 phút** |

> Nếu đã tải ảnh lần trước, script tự bỏ qua file đã tồn tại.

### Kiểm tra thành công

Console in cuối cùng phải có:
```
train :  5000 (2500 real, 2500 fake)
val   :   400 ( 200 real,  200 fake)
test  :   ~498 (200 seen + ~298 OOD)
Subreddits ONLY in test (true OOD): {'neutralnews', 'theonion'}
Đã lưu image_embeddings.npy: shape=(N, 512)
CSV nodes/edges đã tạo xong tại data/processed/
```

### Nếu bị lỗi

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `[LỖI NGHIÊM TRỌNG] CLIP thất bại` | Thiếu `transformers` hoặc `Pillow` | `uv sync` lại |
| `[LỖI NGHIÊM TRỌNG] SentenceTransformer thất bại` | Thiếu `sentence-transformers` | `uv sync` lại |
| `Không tìm thấy multimodal_train.tsv` | File chưa đặt vào thư mục gốc | Đặt file TSV đúng vị trí |

---

## Bước 2 — Nạp vào Neo4j và chạy GDS

**File:** `pipeline/02_neo4j_import.py`

### Bước này làm gì?

```
data/processed/*.csv   (đầu ra Bước 1)
        │
        ▼ 1. Kết nối Neo4j qua Bolt (localhost:7887)
        │
        ▼ 2. Xóa dữ liệu cũ, tạo unique constraints
        │
        ▼ 3. Import nodes theo batch 500:
        │    Post, User, Subreddit, Domain, Image
        │
        ▼ 4. Import edges:
        │    POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE, MEMBER_OF
        │
        ▼ 5. Chạy Neo4j GDS algorithms:
        │    PageRank   → Domain.pagerank        (uy tín domain)
        │    Louvain    → Post.community_id      (cộng đồng bài viết)
        │    Betweenness→ User.betweenness       (ảnh hưởng user)
        │    FastRP 64d → post_fastrp.npy        (embedding đồ thị)
        │    NodeSim    → SIMILAR_TO edges       (user tương đồng)
        │
        ▼ 6. Xuất CSV enriched (có thêm pagerank, community_id, betweenness)
             posts_enriched.csv, users_enriched.csv,
             subreddits_enriched.csv, domains_enriched.csv,
             images_enriched.csv, post_fastrp.npy
```

### Chuẩn bị: Khởi động Neo4j

```powershell
# Chạy Neo4j bằng Docker (chỉ cần lần đầu hoặc sau khi tắt máy)
docker compose up -d

# Kiểm tra đang chạy
docker ps
# Phải thấy: neo4j    0.0.0.0:7887->7687/tcp, 0.0.0.0:7874->7474/tcp
```

Đăng nhập Neo4j Browser (tùy chọn, để xem đồ thị):
- URL: `http://localhost:7874`
- User: `neo4j` / Password: `password123`

### Lệnh chạy

```powershell
$env:PYTHONUTF8 = "1"
uv run python pipeline/02_neo4j_import.py
```

### Thời gian ước tính: 5–10 phút

### Nếu không có Docker

Script tự động fallback sang **NetworkX** (thư viện Python):
- PageRank, Louvain, Betweenness, FastRP tính local
- Kết quả GNN không thay đổi
- Không có Neo4j Browser để xem đồ thị

```
[CẢNH BÁO] Neo4j không kết nối được: ...
Chuyển sang local graph analysis bằng NetworkX...
```

### Kiểm tra thành công

```
Import hoàn tất!
GDS algorithms hoàn tất!
Đã lưu post_fastrp.npy: shape=(N, 64)
Lưu enriched CSV xong!
```

---

## Bước 3 — Train mô hình CausalHeteroGNN

**File:** `pipeline/05_train_gnn.py`

### Bước này làm gì?

```
data/processed/*_enriched.csv + *.npy   (đầu ra Bước 2)
        │
        ▼ 1. build_heterodata():
        │    Đọc CSV + npy → HeteroData tensor cho PyG
        │    Post.x:      (N, 771) = text(768) + scalar(3)
        │    User.x:      (N, 4)
        │    Subreddit.x: (N, 3)
        │    Domain.x:    (N, 3)
        │    Image.x:     (N, 512) = CLIP embedding
        │
        ▼ 2. Khởi tạo CausalHeteroGNN:
        │    Linear projection cho mỗi node type → hidden=96
        │    2 lớp HeteroSAGE (SAGEConv, aggr="sum")
        │    GRL (α=2.0) + confounder_clf
        │    causal_head + spurious_head
        │    4 bộ phân loại MLP (2way+6way × baseline+causal)
        │
        ▼ 3. Training loop (max 300 epochs, patience=30):
        │    Forward pass → 2 nhánh G + G_causal
        │    Tính L_total = L_base + L_causal + L_adv + L_ortho
        │    Backward + clip gradient (max_norm=1.0)
        │    Early stopping theo val loss
        │    Lưu checkpoint tốt nhất → models/causal_gnn.pt
        │
        ▼ 4. Test evaluation:
        │    Transductive: full graph inference
        │    Strict Inductive: cắt cạnh test Post
        │    OOD split: seen vs unseen subreddits
        │    Counterfactual engine (do-calculus)
        │    Causal path attribution (gradient-based)
        │
        ▼ Lưu:
             models/causal_gnn.pt
             results/metrics.json
             results/training_history.json
             results/counterfactuals.json
             results/causal_paths.json
```

### Lệnh chạy cơ bản

```powershell
$env:PYTHONUTF8 = "1"
uv run python pipeline/05_train_gnn.py
```

### Lệnh chạy nhanh (bỏ qua counterfactual để tiết kiệm ~10 phút)

```powershell
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"
uv run python pipeline/05_train_gnn.py
```

### Các biến môi trường quan trọng

| Biến | Mặc định | Ý nghĩa |
|------|---------|---------|
| `GNN_SEED` | `42` | Random seed để tái lập kết quả |
| `GNN_HIDDEN` | `96` | Số chiều embedding ẩn |
| `GNN_DROPOUT` | `0.4` | Tỉ lệ dropout (regularization) |
| `GNN_GRL_ALPHA` | `2.0` | Cường độ adversarial (GRL) |
| `GNN_LR` | `0.005` | Learning rate ban đầu |
| `GNN_SKIP_EXPLAIN` | `0` | `1` = bỏ counterfactual (chạy nhanh hơn) |
| `GNN_RUN_TAG` | `""` | Hậu tố file output (ví dụ `_seed42`) |
| `GNN_CAUSAL_CUT` | `1` | `0` = tắt backdoor cut (ablation study) |
| `GNN_NEUTRAL_DOMAIN` | `0` | `1` = trung hòa fake_ratio domain (ablation) |
| `GNN_USE_FASTRP` | `0` | `1` = dùng FastRP (chỉ cho ablation leakage) |

### Ví dụ chạy với seed khác

```powershell
$env:PYTHONUTF8 = "1"
$env:GNN_SEED = "1"
$env:GNN_RUN_TAG = "_seed1"
$env:GNN_SKIP_EXPLAIN = "1"
uv run python pipeline/05_train_gnn.py
# Output: models/causal_gnn_seed1.pt, results/metrics_seed1.json
```

### Thời gian ước tính

| Giai đoạn | Thời gian |
|-----------|-----------|
| build_heterodata() | ~30 giây |
| Training (300 epochs, early stop) | 2–5 phút |
| Test evaluation | 1–2 phút |
| Counterfactual (nếu bật) | 5–15 phút |
| **Tổng (GNN_SKIP_EXPLAIN=1)** | **~5 phút** |

### Theo dõi trong khi chạy

Mỗi 10 epoch, console in:

```
Epoch 010 | Loss: 2.3451 | Val Loss: 1.8234 | Val F1 Base: 0.7821 | Val F1 Causal: 0.7543 | Patience: 0/30
Epoch 020 | Loss: 1.9823 | Val Loss: 1.5671 | Val F1 Base: 0.8234 | Val F1 Causal: 0.8012 | Patience: 0/30
...
*** Early stopping tại epoch 187 (val loss không cải thiện trong 30 epochs) ***
```

### Kết quả cuối in trên console

```
======================================================================
  Baseline GNN — 2-Way [Strict Inductive]
======================================================================
  Accuracy:        0.7234
  Macro F1:        0.7189
  AUC-ROC:         0.7812
======================================================================
  Causal GNN   — 2-Way [Strict Inductive]
======================================================================
  Accuracy:        0.7512
  Macro F1:        0.7434
  AUC-ROC:         0.8021

OOD GENERALIZATION SUMMARY (Strict Inductive)
  Baseline — Seen F1: 0.7823 | OOD F1: 0.6234 | Drop: 20.33%
  Causal   — Seen F1: 0.7912 | OOD F1: 0.7123 | Drop:  9.98%
  Causal GNN reduces F1 Drop by: 10.35%
```

### Kiểm tra thành công

- File `models/causal_gnn.pt` được tạo
- File `results/metrics.json` được tạo
- Console in "OOD F1 Drop" của Causal GNN nhỏ hơn Baseline

### Nếu bị lỗi

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `FileNotFoundError: posts_enriched.csv` | Chưa chạy Bước 2 | Chạy Bước 2 trước |
| `CUDA out of memory` | GPU quá nhỏ | Thêm `$env:CUDA_VISIBLE_DEVICES=""` để dùng CPU |
| Val F1 = 1.0 từ epoch đầu | Shortcut transductive (bình thường) | Không cần xử lý — early stop dùng val LOSS |

---

## Bước 4 — Đánh giá chi tiết và giải thích nhân quả

**File:** `pipeline/06_evaluate.py`

### Bước này làm gì?

```
models/causal_gnn.pt   (checkpoint từ Bước 3)
        │
        ▼ 1. Load model + build HeteroData (giống Bước 3)
        │
        ▼ 2. Temperature calibration trên val set
        │    Tìm T tối ưu để softmax probability được calibrate tốt
        │    (không thay đổi accuracy, chỉ thay xác suất báo cáo)
        │
        ▼ 3. Inductive inference (strict content-only):
        │    Xóa cạnh của TẤT CẢ test Post → phân loại từ feature riêng
        │    Tính metrics: accuracy, macro F1, AUC-ROC, confusion matrix
        │    Tách seen vs OOD subreddits → F1-Drop
        │
        ▼ 4. Counterfactual Engine (nếu GNN_SKIP_EXPLAIN=0):
        │    CF1: do(image=None)       → xóa ảnh, chạy lại
        │    CF2: do(domain=credible)  → thay domain thành uy tín nhất
        │    CF3: do(subreddit=neutral)→ thay subreddit thành trung tính nhất
        │    Tính Label Flip Rate (LFR)
        │
        ▼ 5. Causal Path Attribution (gradient-based):
        │    Tính gradient của prob_fake theo Subreddit/Domain/Image features
        │    → Đâu là yếu tố quyết định nhất cho mỗi bài?
        │
        ▼ Lưu:
             results/metrics.json        (đè lên file của Bước 3)
             results/counterfactuals.json
             results/causal_paths.json
```

### Lệnh chạy

```powershell
$env:PYTHONUTF8 = "1"
uv run python pipeline/06_evaluate.py
```

### Chạy nhanh (bỏ counterfactual)

```powershell
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"
uv run python pipeline/06_evaluate.py
```

### Thời gian ước tính: 3–20 phút (tùy GNN_SKIP_EXPLAIN)

### Kiểm tra thành công

```
Calibrated temperatures: T_baseline=1.23, T_causal=1.45
[eval mode] INDUCTIVE content-only (leak-free)
Saved metrics.json successfully!
Saved counterfactuals.json successfully!
Saved causal_paths.json successfully!
Evaluation completed successfully!
```

### Ví dụ nội dung causal_paths.json

```json
{
  "abc123": {
    "post_id": "abc123",
    "title": "Scientists Discover New Species Of Bird...",
    "subreddit": {"name": "theonion", "bias": 0.5, "attribution": 0.12},
    "domain": {"name": "theonion.com", "credibility": 0.5, "attribution": 0.61},
    "image": {"attribution": 0.27},
    "confidence": 0.96,
    "explanation": "Flagged primarily due to source credibility. The domain 'theonion.com'..."
  }
}
```

---

## Chạy đầy đủ cả pipeline (1 lệnh)

Nếu muốn chạy tuần tự không cần giám sát:

```powershell
$env:PYTHONUTF8 = "1"
$env:GNN_SKIP_EXPLAIN = "1"   # bỏ counterfactual cho nhanh

# Bước 1: Chuẩn bị dữ liệu (30-70 phút)
uv run python pipeline/01_prepare_data.py

# Bước 2: Neo4j import + GDS (5-10 phút)
# (cần Neo4j đang chạy: docker compose up -d)
uv run python pipeline/02_neo4j_import.py

# Bước 3: Train GNN (5 phút)
uv run python pipeline/05_train_gnn.py

# Bước 4: Evaluate (3 phút)
uv run python pipeline/06_evaluate.py

Write-Host "Pipeline hoàn tất! Xem kết quả tại results/metrics.json"
```

---

## Cấu trúc thư mục đầu ra

Sau khi chạy xong 4 bước:

```
Fakeddit/
├── data/
│   ├── images/                     ← ảnh đã tải (*.jpg)
│   └── processed/
│       ├── sampled_master.csv      ← bảng tổng hợp
│       ├── posts.csv               ← node Post
│       ├── users.csv               ← node User
│       ├── subreddits.csv          ← node Subreddit
│       ├── domains.csv             ← node Domain
│       ├── images.csv              ← node Image (metadata)
│       ├── posted_by.csv           ← cạnh Post→User
│       ├── posted_in.csv           ← cạnh Post→Subreddit
│       ├── links_to.csv            ← cạnh Post→Domain
│       ├── has_image.csv           ← cạnh Post→Image
│       ├── member_of.csv           ← cạnh User→Subreddit
│       ├── post_embeddings.npy     ← (N, 768) SentenceTransformer
│       ├── image_embeddings.npy    ← (N, 512) CLIP
│       ├── posts_enriched.csv      ← Post + community_id (GDS)
│       ├── users_enriched.csv      ← User + betweenness (GDS)
│       ├── domains_enriched.csv    ← Domain + pagerank (GDS)
│       ├── subreddits_enriched.csv
│       ├── images_enriched.csv
│       └── post_fastrp.npy         ← (N, 64) FastRP (không dùng trong GNN)
├── models/
│   └── causal_gnn.pt               ← checkpoint mô hình tốt nhất
└── results/
    ├── metrics.json                 ← kết quả đánh giá (chính)
    ├── training_history.json        ← lịch sử train/val loss theo epoch
    ├── counterfactuals.json         ← do-calculus interventions
    └── causal_paths.json            ← gradient attribution per post
```

---

## Lỗi thường gặp và cách xử lý

| Lỗi | File | Nguyên nhân | Giải pháp |
|-----|------|-------------|-----------|
| `UnicodeEncodeError cp1252` | Bất kỳ | Thiếu PYTHONUTF8 | `$env:PYTHONUTF8="1"` |
| `multimodal_train.tsv not found` | 01 | File TSV chưa có | Đặt file vào thư mục gốc |
| `CLIP thất bại` | 01 | Thiếu transformers/Pillow | `uv sync` |
| `Neo4j không kết nối` | 02 | Docker chưa chạy | `docker compose up -d` hoặc cứ chạy (fallback NetworkX) |
| `posts_enriched.csv not found` | 05 | Chưa chạy Bước 2 | Chạy Bước 2 trước |
| `causal_gnn.pt not found` | 06 | Chưa chạy Bước 3 | Chạy Bước 3 trước |
| `Val F1 = 1.0` ngay epoch 1 | 05 | Shortcut transductive (bình thường) | Không cần xử lý |
| OOD accuracy rất thấp (~50%) | 05/06 | Model chưa hội tụ | Chạy thêm seed (`GNN_SEED=1,2`) và lấy trung bình |
