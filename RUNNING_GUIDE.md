# RUNNING GUIDE — CausalHeteroGNN trên Fakeddit
### Hướng dẫn chạy chi tiết từng bước của toàn bộ pipeline

> Cập nhật: 2026-06-12. Mọi số liệu trong paper đều tái lập được
> bằng các lệnh trong file này. Nguồn số liệu chuẩn: `results/final_tables.md`.
>
> ⚠️ **TÁI CẤU TRÚC (2026-06-12):** toàn bộ script đã chuyển vào **`pipeline/`** và
> **ĐỔI TÊN theo thứ tự chạy 01→17**. File này còn dùng TÊN CŨ — tra bảng đối chiếu
> trong `pipeline/README.md`. Các tên hay gặp: 01_prepare_data (giữ) ·
> 03_train_gnn→**06_train_gnn** · 04_evaluate→**07_evaluate** ·
> 06_baselines_irm_eerm→**08_…** · 07_baselines_erm_mlp→**09_…** ·
> 08_worst_group→**11_…** · 09_final_tables→**12_…** · 05_dashboard→**17_dashboard**.
> Chạy từ thư mục gốc: `uv run python pipeline/06_train_gnn.py`.
> Quy trình chuẩn + script chạy tổng: **`pipeline/README.md`**,
> **`pipeline/run_all.ps1`** (`powershell -File pipeline/run_all.ps1 -Phase all`).

---

## PHẦN A — Ý NGHĨA CỦA NGHIÊN CỨU

### Bài toán
Phát hiện tin giả đa phương thức (text + ảnh) trên mạng xã hội Reddit (bộ dữ liệu
Fakeddit). Dữ liệu được mô hình hóa thành **đồ thị hỗn hợp (HIN)** gồm 5 loại nút:
Post, User, Subreddit, Domain, Image — lưu trong Neo4j và học bằng Heterogeneous
GraphSAGE.

### Vấn đề cốt lõi mà nghiên cứu chỉ ra
GNN thông thường (ERM) **không học nội dung** mà học **lối tắt (shortcut)**: trên
Fakeddit, mỗi cộng đồng (Subreddit) có tỷ lệ tin giả lịch sử rất lệch (có sub gần
100% fake, có sub gần 0%). Model chỉ cần nhìn "bài này đăng ở sub nào" là đoán được
nhãn — đạt ~91% khi test cùng phân phối. Nhưng khi tương quan cộng đồng–nhãn **đảo
ngược** (Confounding-Shift Benchmark, ρ=0.9 → 0.1), ERM sụp xuống 52.1% (AUC ≈ 0.51,
không hơn tung đồng xu). Đây chính là lý do các hệ thống phát hiện tin giả "rất tốt
trên benchmark" nhưng thất bại khi triển khai sang cộng đồng/môi trường mới.

### Giải pháp đề xuất
**CausalHeteroGNN** = can thiệp cấu trúc (structural intervention): nhánh nhân quả
của model encode trên đồ thị đã **cắt toàn bộ cạnh nối tới Subreddit** (chặn đường
cửa sau X ← C₁ → Y), kèm Gradient Reversal Layer + ràng buộc trực giao. Kết quả:
74.2% OOD accuracy khi confounder bị đảo — hơn hẳn các phương pháp phạt mềm
IRM (54.1%) và EERM (59.6%).

### Ba thông điệp khoa học chính (sau revision)
1. **Cắt cứng > phạt mềm**: một khi confounder đã được nhận diện, xóa thẳng đường
   dẫn cấu trúc loại bỏ nó triệt để hơn các ràng buộc bất biến mềm (IRM/EERM) —
   thứ tự nhất quán trên cả 3 seeds: ERM 52.1 < IRM 54.1 < EERM 59.6 < Causal 74.2,
   và worst-group accuracy: Causal 37.8% vs các phương pháp khác ≤ 28.6%.
2. **Đánh giá phải trung thực**: FastRP tính trên toàn đồ thị làm rò rỉ nhãn → OOD
   "ảo" 94%; loại ra thì chỉ còn ~61%. Ablation §5.6 chỉ rõ ~15.4 điểm của 74.2%
   đến từ lịch sử uy tín domain, phần còn lại trùng trần content-only (MLP 59.5%).
3. **Giới hạn thật của bài toán**: với cộng đồng hoàn toàn mới, đồ thị gần như
   không giúp gì so với MLP chỉ dùng nội dung; nhóm khó nhất là nhận diện satire
   (r/theonion: mọi model chỉ 29–44%).

### Giá trị thực tiễn
- Quy trình **kiểm toán shortcut/leakage** cho hệ thống phát hiện tin giả (benchmark
  confounding-shift + ablation + worst-group) dùng được cho bất kỳ pipeline GNN nào.
- Toàn bộ chạy được trên **CPU, dữ liệu 5,898 posts, Neo4j + BI dashboard** —
  triển khai được ở quy mô phòng lab.

---

## PHẦN B — TỔNG QUAN PIPELINE

```
multimodal_train.tsv / multimodal_validate.tsv   (Fakeddit gốc, tải riêng)
        │
[01_prepare_data.py]      tải ảnh, lọc, chia train/val/test + OOD,
        │                 mpnet text-emb (768d), CLIP image-emb (512d),
        │                 feature thống kê train-only → data/processed/*.csv|.npy
        ▼
[docker compose up -d]    Neo4j 5.12 + GDS + APOC (port 7887 bolt / 7874 browser)
[02_neo4j_import.py]      import HIN vào Neo4j, chạy GDS (PageRank, Louvain,
        │                 Betweenness, FastRP, NodeSimilarity) → *_enriched.csv
        ▼
[make_confounded_dataset.py]  sinh benchmark confounding-shift (ρ=0.9→0.1)
        │                     → data/processed_confounded/
        ▼
[03_train_gnn.py]         train CausalHeteroGNN (2 nhánh, GRL, ortho)
[04_evaluate.py]          đánh giá inductive/transductive + counterfactual + LFR
[06_baselines_irm_eerm.py]  IRM + EERM (cùng backbone)
[07_baselines_erm_mlp.py]   ERM độc lập + MLP content-only
[08_worst_group.py]       worst-group accuracy từ checkpoint
        ▼
[09_final_tables.py]      gộp tất cả → results/final_tables.md (bảng paper)
[make_paper_figures.py / make_main_results_dashboard_figure.py]  hình PNG
[05_dashboard.py]         BI dashboard Streamlit
[revise_docx_EN.py / revise_docx_VN.py]  ghi số liệu + hình vào docx paper
```

---

## PHẦN C — CÀI ĐẶT MÔI TRƯỜNG

### C.1. Yêu cầu
| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| Windows + PowerShell | 10/11 | pipeline đã test trên Windows 11 |
| Python | 3.12 (`.python-version`) | quản lý bằng **uv** |
| uv | mới nhất | `pip install uv` hoặc installer chính thức |
| Docker Desktop | bất kỳ | chỉ cần cho Neo4j (bước 02) |
| RAM | ≥ 8 GB | toàn bộ chạy CPU, không cần GPU |
| Ổ đĩa | ≥ 5 GB | ảnh tải về + embeddings |

### C.2. Cài dependencies
```powershell
cd D:\B_Learn_IT_on_Tube\Teacher_PA\DE_TAI_BAO_FAIR\Fakeddit
uv sync          # đọc pyproject.toml + uv.lock, tạo .venv
```
Dependencies chính: torch (CPU), torch-geometric, sentence-transformers,
transformers (CLIP), neo4j, networkx, streamlit, scikit-learn, python-docx.

### C.3. Biến môi trường bắt buộc trên Windows
```powershell
$env:PYTHONUTF8 = "1"
```
**Luôn set trước mọi lệnh** — script in tiếng Việt, console cp1252 sẽ crash nếu thiếu.

### C.4. File `.env` (cho Neo4j)
```
NEO4J_URI=bolt://localhost:7887
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

### C.5. Dữ liệu gốc (tải riêng, không có trong repo)
Đặt 2 file Fakeddit vào thư mục gốc dự án:
- `multimodal_train.tsv` (~155 MB)
- `multimodal_validate.tsv` (~16 MB)

Nguồn: https://github.com/entitize/Fakeddit (bản multimodal).

---

## PHẦN D — CHI TIẾT TỪNG BƯỚC

> ⚠️ **Quy tắc env var**: các script đọc cấu hình qua biến môi trường. Khi đổi
> protocol, **phải xóa biến cũ**, nếu không kết quả sẽ sai âm thầm:
> ```powershell
> Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE, Env:GNN_NEUTRAL_DOMAIN, `
>             Env:GNN_RUN_TAG, Env:GNN_SEED, Env:GNN_SKIP_EXPLAIN -ErrorAction SilentlyContinue
> ```

### BƯỚC 1 — Chuẩn bị dữ liệu: `01_prepare_data.py`

**Làm gì:**
1. Đọc 2 file TSV gốc, lọc bài có ảnh thật (`hasImage`, URL hợp lệ).
2. Tách OOD: `r/neutralnews` (toàn Real) + `r/theonion` (toàn Fake) bị **loại hoàn
   toàn** khỏi train/val — chỉ dùng làm OOD test (~50% fake tổng hợp).
3. Sample cân bằng: train 2500 real + 2500 fake; val 400; seen-test 200; OOD ~300.
4. Tải ảnh song song (15 luồng) về `data/images/`.
5. Trích **text embedding** tiêu đề bằng `all-mpnet-base-v2` (768-d, normalize).
6. Trích **image embedding** bằng CLIP ViT-B/32 (512-d).
   - Nếu CLIP/mpnet lỗi → script **dừng hẳn** (fail-hard, không dùng random).
7. Tính feature thống kê **chỉ từ train** (chống leak): User (post_count, avg_score,
   avg_upvote_ratio, fake_rate), Subreddit (post_count, fake_ratio_real, avg_score),
   Domain (post_count, fake_ratio_real, avg_upvote_ratio). Node chưa thấy trong
   train nhận giá trị trung tính (fake_rate = 0.5).
8. Xuất CSV nodes/edges cho Neo4j + `.npy` embeddings.

**Chạy:**
```powershell
$env:PYTHONUTF8="1"
uv run python 01_prepare_data.py
```
**Thời gian:** 30–90 phút (chủ yếu tải ảnh + encode CLIP trên CPU).

**Output (`data/processed/`):** `sampled_master.csv`, `posts.csv`, `users.csv`,
`subreddits.csv`, `domains.csv`, `images.csv`, `posted_by/posted_in/links_to/`
`has_image/member_of.csv`, `post_embeddings.npy` (N×768),
`image_embeddings.npy` (N×512).

**Kiểm tra đạt:** console in phân phối cuối — train 5000 / val 400 / test ~498
(200 seen + ~298 OOD); dòng `Subreddits ONLY in test (true OOD): {'neutralnews','theonion'}`.

---

### BƯỚC 2 — Neo4j + Graph Data Science: `02_neo4j_import.py`

**Khởi động Neo4j:**
```powershell
docker compose up -d        # Neo4j 5.12 + GDS + APOC
# Browser: http://localhost:7874  (neo4j / password123)
```

**Làm gì:**
1. Xóa dữ liệu cũ, tạo unique constraint cho 5 loại nút.
2. Import nodes + edges theo batch 500 (UNWIND/MERGE).
3. Chạy GDS trên projection undirected:
   - **PageRank** → uy tín Domain (BI)
   - **Louvain** → community_id của Post (BI)
   - **Betweenness** → ảnh hưởng User (BI)
   - **FastRP 64-d** → `post_fastrp.npy` (⚠️ CHỈ dùng cho phân tích leakage §5.4
     và dashboard — mặc định KHÔNG đưa vào GNN)
   - **NodeSimilarity** → cạnh SIMILAR_TO giữa User
4. Lưu metadata Causal DAG (node `:CausalDAG`) vào Neo4j.
5. Xuất `*_enriched.csv` (đã merge kết quả GDS).

**Chạy:**
```powershell
$env:PYTHONUTF8="1"
uv run python 02_neo4j_import.py
```
**Thời gian:** 5–10 phút.
**Không có Docker?** Script tự fallback sang NetworkX (PageRank/Louvain/Betweenness/
FastRP local) — kết quả GNN không đổi, chỉ mất phần BI trên Neo4j.

**Output:** `posts_enriched.csv`, `users_enriched.csv`, `subreddits_enriched.csv`,
`domains_enriched.csv`, `images_enriched.csv`, `post_fastrp.npy`.

---

### BƯỚC 3 — Sinh benchmark Confounding-Shift: `make_confounded_dataset.py`

**Làm gì:** Tạo phiên bản dữ liệu kiểu ColoredMNIST để **đo trực tiếp** khả năng
chống confounder:
- Thay subreddit thật bằng biến nhị phân tổng hợp: `spur_fakebias` / `spur_realbias`.
- Train/val/seen-test: env trùng nhãn với xác suất **ρ = 0.9** (shortcut gần hoàn hảo).
- OOD test (chính là các bài held-out cũ): tương quan **đảo ngược, ρ = 0.1**.
- Nội dung (text/ảnh/scalar) GIỮ NGUYÊN — chỉ subreddit bị thay; tái dùng embeddings.

**Chạy:**
```powershell
$env:PYTHONUTF8="1"
uv run python make_confounded_dataset.py
```
**Thời gian:** < 1 phút.
**Output:** `data/processed_confounded/` (drop-in thay `data/processed` qua
`GNN_INPUT_DIR`). Console in `P(env==fake): train≈0.90 | seen-test≈0.90 | OOD-test≈0.10`.

---

### BƯỚC 4 — Train CausalHeteroGNN: `03_train_gnn.py`

**Kiến trúc (khớp Hình 2 paper):**
- Mỗi loại nút chiếu Linear → 96-d, ReLU, dropout 0.4.
- 2 lớp HeteroSAGE (SAGEConv mỗi loại quan hệ, aggr="sum") — **encoder dùng chung**.
- **Nhánh baseline**: encode đồ thị đầy đủ G.
- **Nhánh nhân quả**: encode G_causal = G \ {mọi cạnh chạm Subreddit} (backdoor cut).
- GRL (α=2.0) trên h_c → confounder classifier (đoán subreddit-ID, gradient đảo).
- Ràng buộc trực giao |cos(h_c, h_s)|.
- Loss: `L = L_base,2w + 0.5·L_base,6w + L_causal,2w + 0.5·L_causal,6w
  + 0.5·L_spurious + 0.5·L_adv + 0.2·L_ortho`.
- Early stopping theo val loss (patience 30, max 300 epochs), gradient clipping,
  class weights 6-way, ReduceLROnPlateau.

**Bảng env var:**
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GNN_SEED` | 42 | random seed |
| `GNN_RUN_TAG` | "" | hậu tố file output (vd `_main_s42`) |
| `GNN_INPUT_DIR` | data/processed | đổi sang `data/processed_confounded` cho benchmark |
| `GNN_USE_FASTRP` | 0 | 1 = đưa FastRP vào input (CHỈ cho ablation leakage §5.4) |
| `GNN_CAUSAL_CUT` | 1 | 0 = tắt backdoor cut (ablation) |
| `GNN_NEUTRAL_DOMAIN` | 0 | 1 = domain fake_ratio → 0.5 (ablation §5.6) |
| `GNN_NEUTRAL_USER` | 0 | 1 = user fake_rate → 0.5 |
| `GNN_SKIP_EXPLAIN` | 0 | 1 = bỏ counterfactual/LFR (nhanh hơn nhiều) |
| `GNN_HIDDEN/DROPOUT/GRL_ALPHA/LR/WD/W_ADV` | 96/0.4/2.0/0.005/5e-4/0.5 | hyperparams |

**Chạy (1 seed, protocol chuẩn):**
```powershell
$env:PYTHONUTF8="1"; $env:GNN_SEED="42"; $env:GNN_RUN_TAG="_main_s42"; $env:GNN_SKIP_EXPLAIN="1"
uv run python 03_train_gnn.py
```
**Thời gian:** ~2–5 phút/seed trên CPU.
**Output:** `models/causal_gnn_main_s42.pt`, `results/metrics_main_s42.json`,
`results/training_history.json`.

---

### BƯỚC 5 — Đánh giá chuẩn: `04_evaluate.py`

**Làm gì:** Load checkpoint và đánh giá lại theo đúng chế độ của từng protocol
(**file metrics cuối cùng của paper do bước này ghi**, đè lên metrics của 03):
- **Mặc định (standard OOD)**: *inductive content-only* — mask MỌI cạnh chạm bài
  test (2 chiều); mỗi bài test chỉ được phân loại từ feature của chính nó.
- **`GNN_OOD_TRANSDUCTIVE=1` (confounding-shift)**: giữ nguyên message passing —
  bắt buộc, vì mục đích benchmark là cho model "nhìn thấy" confounder bị đảo.
- Temperature calibration trên val (không đụng test).
- Counterfactual engine: do(image=∅), do(domain=credible), do(subreddit=neutral)
  → `counterfactuals.json`, LFR; gradient attribution → `causal_paths.json`
  (bỏ qua nếu `GNN_SKIP_EXPLAIN=1`).

**Chạy:**
```powershell
# tiếp nối bước 4, cùng RUN_TAG:
uv run python 04_evaluate.py
```

---

### BƯỚC 6 — Baseline IRM + EERM: `06_baselines_irm_eerm.py`

**Làm gì:** Train 2 baseline phạt-mềm ĐỘC LẬP, cùng backbone HeteroSAGE,
không cut/GRL/ortho:
- **IRM** (IRMv1, λ=100, anneal epoch 50): môi trường = phân hoạch subreddit.
- **EERM** (K=3 env ảo bằng nhiễu cạnh ≤ 0.3, phạt phương sai rủi ro) — biến thể
  thực dụng của REINFORCE edge-editor.

```powershell
$env:PYTHONUTF8="1"; $env:GNN_SEED="42"; $env:GNN_RUN_TAG="_stdood_s42"
uv run python 06_baselines_irm_eerm.py
```
**Output:** `results/baselines_irm_eerm_stdood_s42.json`, `models/irm*.pt`, `models/eerm*.pt`.

---

### BƯỚC 7 — Baseline ERM + MLP: `07_baselines_erm_mlp.py`

**Làm gì (THÊM trong bản revision — quan trọng):**
- **ERM**: HeteroSAGE thuần, train độc lập → đây mới là "Baseline GNN" đúng nghĩa
  của paper (số cũ 36.4% là nhánh baseline joint-trained bên trong CausalHeteroGNN,
  giờ chỉ dùng làm diagnostic).
- **MLP content-only**: phân loại trực tiếp trên feature Post, không đồ thị —
  mốc neo "trần nội dung".

```powershell
$env:PYTHONUTF8="1"; $env:GNN_SEED="42"; $env:GNN_RUN_TAG="_stdood_s42"
uv run python 07_baselines_erm_mlp.py
```
**Output:** `results/baselines_erm_mlp_stdood_s42.json`, `models/erm*.pt`, `models/mlp*.pt`.

---

### BƯỚC 8 — Worst-Group Accuracy: `08_worst_group.py`

**Làm gì:** Load mọi checkpoint có sẵn (causal, ERM, MLP, IRM, EERM × 3 seeds),
tính accuracy theo nhóm (subreddit × nhãn) trên OOD test:
- Standard OOD: nhóm = `neutralnews|Real`, `theonion|Fake`.
- Conf-shift: 4 nhóm env × label; nhóm khó nhất = bị đảo tương quan.

```powershell
# Standard OOD
$env:PYTHONUTF8="1"; uv run python 08_worst_group.py
# Confounding-shift
$env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
uv run python 08_worst_group.py
```
**Output:** `results/worst_group_stdood.json`, `results/worst_group_conf.json`.

---

### BƯỚC 9 — Bảng kết quả cuối: `09_final_tables.py`

```powershell
$env:PYTHONUTF8="1"
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE -ErrorAction SilentlyContinue
uv run python 09_final_tables.py
```
**Output:** `results/final_tables.md` + `final_tables.json` — mean±std 3 seeds cho
mọi model × 2 protocol + ablation + worst-group. **Đây là nguồn số liệu của paper.**

---

## PHẦN E — CÔNG THỨC TÁI LẬP ĐẦY ĐỦ KẾT QUẢ PAPER (3 seeds)

> Chạy tuần tự sau khi xong Bước 1–3. Tổng thời gian ~1–2 giờ CPU.

### E.1. Standard OOD (Table 1 cột Held-Out) — inductive content-only
```powershell
$env:PYTHONUTF8="1"; $env:GNN_SKIP_EXPLAIN="1"
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE -ErrorAction SilentlyContinue
foreach ($s in @("42","1","2")) {
  $env:GNN_SEED=$s
  $env:GNN_RUN_TAG="_main_s$s";   uv run python 03_train_gnn.py; uv run python 04_evaluate.py
  $env:GNN_RUN_TAG="_stdood_s$s"; uv run python 06_baselines_irm_eerm.py
                                  uv run python 07_baselines_erm_mlp.py
}
```

### E.2. Confounding-Shift (Table 1 các cột Conf-Shift) — transductive
```powershell
$env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
foreach ($s in @("42","1","2")) {
  $env:GNN_SEED=$s
  $env:GNN_RUN_TAG="_bd_s$s";   uv run python 03_train_gnn.py; uv run python 04_evaluate.py
  $env:GNN_RUN_TAG="_conf_s$s"; uv run python 06_baselines_irm_eerm.py
                                uv run python 07_baselines_erm_mlp.py
}
```

### E.3. Ablation lịch sử domain (§5.6)
```powershell
$env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"; $env:GNN_NEUTRAL_DOMAIN="1"
foreach ($s in @("42","1","2")) {
  $env:GNN_SEED=$s; $env:GNN_RUN_TAG="_nd_s$s"
  uv run python 03_train_gnn.py; uv run python 04_evaluate.py
}
Remove-Item Env:GNN_NEUTRAL_DOMAIN
```

### E.4. Ablation FastRP leakage (§5.4)
```powershell
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE -ErrorAction SilentlyContinue
$env:GNN_USE_FASTRP="1"
foreach ($s in @("42","1","2")) {
  $env:GNN_SEED=$s; $env:GNN_RUN_TAG="_s$s"
  uv run python 03_train_gnn.py; uv run python 04_evaluate.py
}
Remove-Item Env:GNN_USE_FASTRP
```

### E.5. Worst-group + bảng cuối + LFR
```powershell
uv run python 08_worst_group.py
$env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
uv run python 08_worst_group.py
Remove-Item Env:GNN_INPUT_DIR, Env:GNN_OOD_TRANSDUCTIVE
uv run python 09_final_tables.py
# LFR (Table 2): chạy lại 04 KHÔNG có GNN_SKIP_EXPLAIN với tag _bd_s42
```

### E.6. Mapping bảng paper ↔ file kết quả
| Vị trí trong paper | File nguồn |
|---|---|
| Table 1 (toàn bộ) | `results/final_tables.md` |
| — CausalHeteroGNN | `metrics_main_s*` (held-out) / `metrics_bd_s*` (conf) |
| — Baseline GNN (ERM), MLP | `baselines_erm_mlp_{stdood,conf}_s*` |
| — IRM, EERM | `baselines_irm_eerm_{stdood,conf}_s*` |
| §5.3 diagnostic 36.4%/AUC 0.314 | `metrics_bd_s*` key `baseline` |
| §5.4 FastRP 94% vs 61% | `metrics_s*` (leaky) vs `metrics_nofrp_s*` |
| Table 2 LFR | `metrics_bd_s42.json` key `lfr` + `counterfactuals.json` |
| §5.6 ablation 74.2→58.8 | `metrics_nd_s*` |
| Worst-group | `worst_group_{stdood,conf}.json` |

---

## PHẦN F — HÌNH, DASHBOARD, PAPER

### F.1. Sinh hình cho paper
```powershell
$env:PYTHONUTF8="1"
uv run python make_paper_figures.py                      # fig1..fig7 (Hình 3 = fig2_standard_vs_shift)
uv run python make_main_results_dashboard_figure.py      # Hình 4 (dashboard)
# Output: Fair_Article/figures/ + Fair_Article_VN/figures/
```
Lưu ý: 2 script này đã được patch để "Baseline GNN" = ERM độc lập + đường trần MLP.

### F.2. BI Dashboard (Streamlit)
```powershell
$env:PYTHONUTF8="1"
uv run streamlit run 05_dashboard.py
# mở http://localhost:8501 — cần results/metrics.json, counterfactuals.json,
# causal_paths.json và (tùy chọn) Neo4j đang chạy cho các panel đồ thị
```

### F.3. Ghi số liệu + hình vào paper docx
```powershell
uv run python revise_docx_EN.py   # sửa bản EN (tự backup *_BACKUP_prerevision.docx)
uv run python revise_docx_VN.py   # sửa bản VN
```
Hai script này thay text/bảng/hình theo `RESULTS_UPDATE.md`. Chỉ chạy lại nếu
số liệu thay đổi (script thay theo anchor — chạy 2 lần trên cùng file sẽ báo
lỗi anchor, đó là chủ ý để tránh ghi đè chồng).

---

## PHẦN G — TROUBLESHOOTING

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `UnicodeEncodeError ... cp1252` | Quên `$env:PYTHONUTF8="1"` |
| `Python was not found` | Dùng `uv run python ...`, không gọi `python` trực tiếp |
| Kết quả conf-shift ~17% hoặc lệch hẳn | Env var protocol cũ còn dính — chạy khối `Remove-Item Env:...` ở đầu Phần D |
| `Model checkpoint not found` khi chạy 04 | Chưa chạy 03 với cùng `GNN_RUN_TAG` |
| Neo4j không kết nối được | `docker compose up -d`; kiểm tra port 7887; hoặc cứ chạy — script fallback NetworkX |
| 01 dừng với lỗi CLIP/SentenceTransformer | Fail-hard chủ ý (chống random embedding). Cài lại `transformers`, `sentence-transformers`, `pillow` rồi chạy lại |
| Val F1 = 1.0 ngay từ epoch đầu (ERM/transductive) | Bình thường — shortcut subreddit trong setting transductive; early stop dùng val LOSS |
| OOD acc dao động mạnh giữa seeds (±8) | OOD set chỉ 298 bài — luôn báo cáo mean±std 3 seeds, không lấy 1 seed |
| Số trong docx không khớp `final_tables.md` | docx được sửa thủ công sau đó, hoặc chưa chạy lại revise_docx — đối chiếu `RESULTS_UPDATE.md` |

---

## PHẦN H — CẤU TRÚC FILE QUAN TRỌNG

```
Fakeddit/
├── 01_prepare_data.py … 09_final_tables.py   # pipeline chính (đánh số theo thứ tự chạy)
├── make_confounded_dataset.py                # benchmark ρ=0.9→0.1
├── make_paper_figures.py / make_main_results_dashboard_figure.py
├── revise_docx_EN.py / revise_docx_VN.py     # ghi revision vào paper
├── docker-compose.yml / .env                 # Neo4j 5.12 + GDS + APOC
├── data/processed/                            # dataset chuẩn
├── data/processed_confounded/                 # dataset confounding-shift
├── models/*.pt                                # checkpoints (causal_gnn_*, erm_*, mlp_*, irm_*, eerm_*)
├── results/                                   # mọi metrics JSON + final_tables.md
├── REVIEW_FAIR.md                             # bản phản biện đầy đủ
├── RESULTS_UPDATE.md                          # số liệu chốt + checklist revision
└── (EN|VN)_Causal Graph Disentanglement....docx   # paper (đã revision)
```
