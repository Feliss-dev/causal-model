# FILES — Scripts sinh hình minh họa cho bài báo

Thư mục này chứa 5 script Python (matplotlib) tạo ra 5 hình dùng trong bài báo CausalHeteroGNN.

**Chạy từ thư mục gốc dự án** (nơi có `results/`, `figures/`):
```powershell
uv run python pipeline/11_generate_figures.py   # sinh tất cả 5 hình cùng lúc
```
hoặc chạy từng script riêng (xem bên dưới).

---

## Bản đồ script → hình

| Script | Hình trong bài | Output | Nội dung | Nguồn data |
|---|---|---|---|---|
| `fig1_neo4j_schema.py` | Hình 1 | `figures/image1.png` | Sơ đồ property-graph HIN (Neo4j) | Hardcoded (kiến trúc) |
| `fig2_architecture.py` | Hình 2 | `figures/image2.png` | Kiến trúc CausalHeteroGNN | Hardcoded (kiến trúc) |
| `fig3_ood_performance.py` | Hình 3 | `figures/image3.png` | Hiệu năng 2 giao thức OOD | `results/final_tables.json` |
| `fig4_robustness.py` | Hình 4 | `figures/image5.png` | Worst-Group Acc & Label-Flip Rate | `results/worst_group_conf.json` + `results/metrics_bd_s42.json` |
| `fig5_dashboard.py` | Hình 5 | `figures/image6.png` | BI Dashboard 6 panel | 3 file JSON (xem dưới) |

> **Lưu ý tên file ảnh:** `image5.png` và `image6.png` (không phải image4/image5)  
> vì file .docx đã xóa hình LOCO cũ là image4 khi đánh số lại.

---

## Yêu cầu

```bash
pip install matplotlib numpy
```

**Đối với `fig3`, `fig4`, `fig5`** — cần chạy pipeline trước để có JSON kết quả:

| File cần có | Sinh bởi pipeline |
|---|---|
| `results/final_tables.json` | `pipeline/10_final_tables.py` |
| `results/worst_group_conf.json` | `pipeline/09_worst_group.py` (Conf-Shift protocol) |
| `results/metrics_bd_s42.json` với trường `lfr` | `pipeline/06_evaluate.py`, seed 42, `GNN_SKIP_EXPLAIN=0` |

`fig1` và `fig2` chạy được ngay không cần data pipeline.

---

## Cách chạy từng script

Chạy từ **thư mục gốc dự án**:

```bash
# Hình 1 — sơ đồ Neo4j schema (không cần data)
python files/fig1_neo4j_schema.py

# Hình 2 — kiến trúc model (không cần data)
python files/fig2_architecture.py

# Hình 3 — OOD performance (cần final_tables.json)
python files/fig3_ood_performance.py

# Hình 4 — robustness (cần worst_group_conf.json + metrics_bd_s42.json)
python files/fig4_robustness.py

# Hình 5 — dashboard (cần cả 3 JSON)
python files/fig5_dashboard.py
```

---

## Mô tả chi tiết từng hình

### Hình 3 — `fig3_ood_performance.py`

Hai panel so sánh 5 models (MLP, ERM/Baseline GNN, IRM, EERM, CausalHeteroGNN):
- **Panel (a):** Grouped bar chart 3 nhóm: Seen Accuracy / Held-Out OOD / Confounding-Shift OOD
- **Panel (b):** AUC bars + F1-drop line trên Confounding-Shift

Đọc từ `results/final_tables.json` → `standard_ood` (cho Seen + Held-Out) và `confounding_shift` (cho Conf-Shift, AUC, F1-drop).

### Hình 4 — `fig4_robustness.py`

Hai panel phân tích robustness:
- **Panel (a):** Worst-Group vs Avg-Group Accuracy (Confounding-Shift) — đọc từ `worst_group_conf.json`
- **Panel (b):** Label-Flip Rate dưới 3 can thiệp nhân quả: do(C₁=swap), do(I=∅), do(D=credible)
  - Đọc từ `metrics_bd_s42.json` → trường `lfr.baseline` và `lfr.causal`
  - Fallback về hằng số nếu file chưa có trường `lfr` (khi chạy với `GNN_SKIP_EXPLAIN=1`)

### Hình 5 — `fig5_dashboard.py`

BI Dashboard 6 panel:
1. **KPI cards (5 cards):** Conf-Shift OOD Acc, AUC, F1-drop, Worst-Group, LFR Subreddit
2. **Bar chart:** Confounding-Shift OOD Accuracy (5 models)
3. **Dual-axis chart:** AUC (bars) + F1-drop (line)
4. **Horizontal bar:** Worst-Group Accuracy
5. **Grouped bars (2 cột):** Label-Flip Rate under interventions
6. **Radar chart:** So sánh top-3 models trên 4 KPI (CausalHeteroGNN, EERM, MLP)

---

## Format JSON input

### `results/final_tables.json`
```json
{
  "standard_ood": [
    {
      "model": "MLP (content-only)",
      "seen_acc": [0.817, 0.006],
      "ood_acc":  [0.605, 0.011],
      "ood_auc":  [0.685, 0.020],
      "f1_drop":  [30.0,  1.5]
    },
    { "model": "ERM HeteroSAGE", ... },
    { "model": "IRM", ... },
    { "model": "EERM", ... },
    { "model": "BaselineBranch*", ... },
    { "model": "CausalHeteroGNN", ... }
  ],
  "confounding_shift": [ ... cùng cấu trúc ... ]
}
```
*(accuracy/auc dạng 0–1; f1_drop đã là %, ví dụ 30.0 = 30%)*

### `results/worst_group_conf.json`
```json
{
  "summary": {
    "MLP":             { "worst_group_acc": [0.156, 0.031], "avg_group_acc": [0.565, 0.010] },
    "ERM":             { "worst_group_acc": [0.235, 0.079], "avg_group_acc": [0.732, 0.044] },
    "IRM":             { "worst_group_acc": [0.244, 0.071], "avg_group_acc": [0.743, 0.027] },
    "EERM":            { "worst_group_acc": [0.286, 0.013], "avg_group_acc": [0.770, 0.010] },
    "CausalHeteroGNN": { "worst_group_acc": [0.378, 0.113], "avg_group_acc": [0.708, 0.045] }
  }
}
```
*(giá trị 0–1, ×100 để ra %)*

### `results/metrics_bd_s42.json` (phần LFR)
```json
{
  "lfr": {
    "baseline": { "lfr_subreddit": 0.301, "lfr_image": 0.040, "lfr_domain": 0.062 },
    "causal":   { "lfr_subreddit": 0.000, "lfr_image": 0.080, "lfr_domain": 0.155 }
  }
}
```
*(giá trị 0–1 = tỷ lệ; ×100 để ra %; chỉ có khi chạy `06_evaluate.py` với `GNN_SKIP_EXPLAIN=0`)*
