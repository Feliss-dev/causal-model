# SETUP — Hướng dẫn cài đặt và chạy trên máy mới

Tài liệu này hướng dẫn từng bước để clone repo và chạy toàn bộ pipeline
**CausalHeteroGNN Fakeddit** trên một máy tính mới từ đầu.

---

## Yêu cầu phần cứng

| Tài nguyên | Tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | 4 nhân | 8 nhân+ |
| RAM | 16 GB | 32 GB |
| Ổ đĩa | 30 GB trống | 50 GB trống |
| GPU (CUDA) | không bắt buộc | 6 GB VRAM+ (nhanh hơn ~4–8×) |
| Internet | cần để tải ảnh & model | tốc độ cao = nhanh hơn |

---

## Yêu cầu phần mềm

| Phần mềm | Phiên bản | Ghi chú |
|---|---|---|
| Python | **3.12+** | bắt buộc |
| uv | mới nhất | package manager — cài nhanh hơn pip |
| Git | mới nhất | để clone repo |
| Docker Desktop | mới nhất | **tùy chọn** — chỉ cần cho Neo4j |
| PowerShell | 7+ | để chạy `run_all.ps1` |

---

## Bước 1 — Clone repository

```powershell
git clone <URL_REPO> Fakeddit
cd Fakeddit
```

Sau khi clone, cấu trúc thư mục gốc sẽ là:
```
Fakeddit/
├── pipeline/       ← 12 scripts chính
├── files/          ← 5 scripts vẽ hình
├── pyproject.toml  ← dependencies
├── docker-compose.yml
├── SETUP.md        ← file này
└── ...
```

---

## Bước 2 — Cài đặt Python environment

### Cách A — Dùng `uv` (khuyến nghị, nhanh hơn)

```powershell
# Cài uv nếu chưa có
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Tạo virtual env và cài tất cả dependencies
uv sync

# Kiểm tra
uv run python --version   # phải là Python 3.12+
```

### Cách B — Dùng `pip` thông thường

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
# Nếu có GPU CUDA 12.x:
# pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu121

pip install torch-geometric==2.7.0
pip install -r requirements.txt
```

### Cài đặt PyTorch Geometric (bắt buộc)

PyTorch Geometric cần cài riêng các extension C++:

```powershell
# Sau khi cài torch:
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
# Thay +cpu bằng +cu121 nếu dùng GPU
```

### Kiểm tra cài đặt

```powershell
uv run python -c "import torch; import torch_geometric; import sentence_transformers; print('OK')"
# Phải in ra: OK
```

---

## Bước 3 — Tải dữ liệu Fakeddit

Pipeline cần 2 file TSV gốc đặt **tại thư mục gốc** (cùng cấp với `pipeline/`):

| File | Kích thước | Nguồn |
|---|---|---|
| `multimodal_train.tsv` | ~500 MB | Fakeddit dataset |
| `multimodal_validate.tsv` | ~50 MB | Fakeddit dataset |
| `multimodal_test_public.tsv` | ~50 MB | tùy chọn |

**Tải từ Fakeddit GitHub:**
```
https://github.com/entitize/Fakeddit
```

Đặt file vào thư mục gốc dự án:
```
Fakeddit/
├── multimodal_train.tsv       ← đặt ở đây
├── multimodal_validate.tsv    ← đặt ở đây
└── pipeline/
```

> **Lưu ý:** Ảnh sẽ được tải tự động bởi `01_prepare_data.py` từ URL trong TSV
> (cần Internet, ~5–10 GB, mất 1–3 giờ tùy tốc độ mạng).

---

## Bước 4 — Cài đặt Neo4j (tùy chọn)

Neo4j dùng để lưu knowledge graph và chạy GDS algorithms (PageRank, Louvain, Betweenness).
**Nếu không cài, pipeline tự fallback sang NetworkX** — kết quả GNN không đổi.

### Cách A — Docker (khuyến nghị)

```powershell
# Đảm bảo Docker Desktop đang chạy
docker compose up -d

# Kiểm tra Neo4j đã sẵn sàng (chờ ~30 giây)
Start-Sleep 30
Invoke-WebRequest http://localhost:7874 -UseBasicParsing | Select-Object StatusCode
# Phải trả về StatusCode 200
```

Truy cập Neo4j Browser: `http://localhost:7874`
- Username: `neo4j`
- Password: `password123`

Cổng kết nối: `bolt://localhost:7887`

### Cách B — Neo4j standalone (không Docker)

1. Tải Neo4j Community 5.x từ [neo4j.com/download](https://neo4j.com/download/)
2. Cài plugin **Graph Data Science** (GDS)
3. Đặt password = `password123`, cổng Bolt = `7687`
4. Đổi biến môi trường:

```powershell
$env:NEO4J_URI      = "bolt://localhost:7687"
$env:NEO4J_USER     = "neo4j"
$env:NEO4J_PASSWORD = "your_password"
```

### Không dùng Neo4j (fallback NetworkX)

Nếu không có Docker/Neo4j, script `02_neo4j_import.py` tự động dùng NetworkX.
Không cần làm gì thêm — chỉ bỏ qua Bước 4.

---

## Bước 5 — Chạy pipeline

> **Quan trọng:** Luôn chạy từ **thư mục gốc dự án** (`Fakeddit/`), không phải từ `pipeline/`.

```powershell
cd Fakeddit
$env:PYTHONUTF8 = "1"    # bắt buộc trên Windows để tránh lỗi Unicode
```

### Phase A — Chuẩn bị dữ liệu (chạy 1 lần, ~2–4 giờ)

```powershell
powershell -File pipeline/run_all.ps1 -Phase data
```

Quá trình:
1. `01_prepare_data.py` — tải ảnh (~5 GB), tính CLIP + mpnet embeddings → `data/processed/`
2. `02_neo4j_import.py` — import graph, tính GDS features → `data/processed/*_enriched.csv`
3. `03_clip_consistency.py` — tính điểm nhất quán text-image → `data/processed/clip_cons.npy`
4. `04_make_confounded.py` — tạo benchmark confounding-shift → `data/processed_confounded/`

Kết thúc phase A, kiểm tra:
```powershell
Test-Path data/processed/posts_enriched.csv      # True
Test-Path data/processed/post_embeddings.npy     # True
Test-Path data/processed_confounded/posts_enriched.csv  # True
```

### Phase B — Huấn luyện & đánh giá (3 seeds × 2 protocols, ~2–6 giờ)

```powershell
powershell -File pipeline/run_all.ps1 -Phase main
```

Quá trình (24 lần gọi script):
- 3 seeds (42, 1, 2) × Held-Out protocol: train CausalHeteroGNN + eval + 2 baselines
- 3 seeds × Confounding-Shift protocol: tương tự, thêm LFR run seed 42
- Lưu kết quả vào `results/metrics_*.json` và `models/causal_gnn_*.pt`

Kết thúc phase B, kiểm tra:
```powershell
(Get-ChildItem results -Filter "*.json").Count   # phải >= 12
(Get-ChildItem models -Filter "*.pt").Count      # phải >= 6
```

### Phase C — Tổng hợp & sinh hình (< 5 phút)

```powershell
powershell -File pipeline/run_all.ps1 -Phase aggregate
```

Quá trình:
1. `09_worst_group.py` — tính worst-group accuracy từ checkpoint
2. `10_final_tables.py` — gộp tất cả → `results/final_tables.md`
3. `11_generate_figures.py` — sinh 5 hình → `figures/image*.png`

Kết thúc phase C, kiểm tra:
```powershell
Get-Content results/final_tables.md | Select-Object -First 20
Get-ChildItem figures -Filter "*.png" | Select-Object Name
# Phải thấy: image1.png  image2.png  image3.png  image5.png  image6.png
```

### Hoặc chạy tất cả cùng lúc

```powershell
powershell -File pipeline/run_all.ps1 -Phase all
```

---

## Bước 6 — Xem kết quả

### Bảng số liệu paper
```powershell
Get-Content results/final_tables.md
```

### Hình trong bài báo
Mở thư mục `figures/`:
- `image1.png` — Hình 1: sơ đồ Neo4j schema
- `image2.png` — Hình 2: kiến trúc CausalHeteroGNN
- `image3.png` — Hình 3: hiệu năng OOD
- `image5.png` — Hình 4: worst-group + LFR
- `image6.png` — Hình 5: BI dashboard

### Dashboard tương tác (Streamlit)
```powershell
powershell -File pipeline/run_all.ps1 -Phase dashboard
# Mở trình duyệt: http://localhost:8501
```

---

## Chạy nhanh để kiểm tra (1 seed, không tải ảnh)

Nếu bạn chỉ muốn kiểm tra pipeline hoạt động mà chưa có dữ liệu đầy đủ:

```powershell
$env:PYTHONUTF8 = "1"
$env:GNN_SEED = "42"
$env:GNN_SKIP_EXPLAIN = "1"

# Giả sử đã có data/processed/ từ lần chạy trước
$env:GNN_RUN_TAG = "_main_s42"
uv run python pipeline/05_train_gnn.py
uv run python pipeline/06_evaluate.py
uv run python pipeline/09_worst_group.py
uv run python pipeline/10_final_tables.py
uv run python pipeline/11_generate_figures.py
```

---

## Cấu trúc thư mục sau khi chạy xong

```
Fakeddit/
├── data/
│   ├── processed/              ← node/edge CSVs + embeddings
│   ├── processed_confounded/   ← benchmark confounding-shift
│   └── images/                 ← ảnh JPG đã tải (~5 GB)
├── models/
│   ├── causal_gnn_main_s42.pt
│   ├── causal_gnn_main_s1.pt
│   ├── causal_gnn_main_s2.pt
│   └── ...                     ← 6+ checkpoints
├── results/
│   ├── metrics_main_s42.json
│   ├── metrics_bd_s42.json     ← có trường "lfr"
│   ├── baselines_*.json
│   ├── worst_group_stdood.json
│   ├── worst_group_conf.json
│   ├── final_tables.md         ← ← nguồn số liệu paper
│   └── final_tables.json
└── figures/
    ├── image1.png              ← Hình 1
    ├── image2.png              ← Hình 2
    ├── image3.png              ← Hình 3
    ├── image5.png              ← Hình 4
    └── image6.png              ← Hình 5
```

---

## Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'torch_geometric'`
```powershell
uv add torch-geometric
# hoặc
pip install torch-geometric==2.7.0
```

### Lỗi: `ModuleNotFoundError: No module named 'torch_scatter'`
```powershell
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.12.0+cpu.html
```

### Lỗi: `UnicodeDecodeError` khi đọc TSV
```powershell
$env:PYTHONUTF8 = "1"   # đặt trước khi chạy bất kỳ script nào
```

### Lỗi: `ConnectionRefusedError` khi kết nối Neo4j
Script sẽ tự fallback NetworkX. Nếu muốn dùng Neo4j:
```powershell
docker compose up -d
Start-Sleep 30    # chờ Neo4j khởi động
uv run python pipeline/02_neo4j_import.py
```

### Lỗi: `FileNotFoundError: results/final_tables.json` khi chạy hình
Cần chạy phase B (training) trước phase C:
```powershell
powershell -File pipeline/run_all.ps1 -Phase main
powershell -File pipeline/run_all.ps1 -Phase aggregate
```

### Chạy trên Linux/macOS

Thay thế lệnh PowerShell bằng:
```bash
export PYTHONUTF8=1
python pipeline/run_all.py    # hoặc chạy từng script
# Hoặc dùng bash wrapper tương đương
```

Các biến môi trường đặt bằng `export` thay vì `$env:`:
```bash
export GNN_SEED=42
export GNN_RUN_TAG=_main_s42
export GNN_SKIP_EXPLAIN=1
uv run python pipeline/05_train_gnn.py
```

### Hết RAM khi training

Giảm kích thước batch bằng cách giảm số node:
```powershell
# Trong 01_prepare_data.py, sửa:
# TRAIN_PER_CLASS = 1000   (mặc định 2500)
# VAL_PER_CLASS = 100      (mặc định 200)
```

Hoặc dùng CPU với batch nhỏ hơn:
```powershell
$env:GNN_HIDDEN = "64"    # mặc định 96
```

### Tải ảnh chậm hoặc lỗi

Script `01_prepare_data.py` dùng 10 luồng song song. Có thể giảm:
```python
# Trong 01_prepare_data.py, sửa dòng:
MAX_WORKERS = 5    # mặc định 10
```

Ảnh đã tải được cache trong `data/images/` — chạy lại sẽ bỏ qua ảnh đã có.

---

## Tóm tắt lệnh nhanh (copy-paste)

```powershell
# 1. Clone và vào thư mục
git clone <URL> Fakeddit && cd Fakeddit

# 2. Cài môi trường
uv sync

# 3. Đặt file TSV vào thư mục gốc (tải thủ công từ Fakeddit GitHub)

# 4. Khởi động Neo4j (tùy chọn)
docker compose up -d && Start-Sleep 30

# 5. Chạy pipeline
$env:PYTHONUTF8 = "1"
powershell -File pipeline/run_all.ps1 -Phase all

# 6. Xem kết quả
Get-Content results/final_tables.md
Get-ChildItem figures -Filter "*.png"
```

---

## Tài liệu liên quan

- [`pipeline/README.md`](pipeline/README.md) — chi tiết từng script và env-var
- [`files/README.md`](files/README.md) — hướng dẫn chạy riêng scripts vẽ hình
- [`RUNNING_GUIDE.md`](RUNNING_GUIDE.md) — hướng dẫn nâng cao + troubleshooting
