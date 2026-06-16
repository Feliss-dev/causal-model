# Tóm tắt thay đổi — CausalHeteroGNN Paper (VN.md)
**Ngày thực hiện: 2026-06-13**

---

## 1. Số liệu đã cập nhật

### Bảng 3 — Worst-Group Accuracy (Confounding-Shift)
| Trường | Cũ (sai) | Mới (đúng, từ final_tables.md) |
|---|---|---|
| CausalHeteroGNN Worst-Group Acc | 56.0±21.4 | **37.8±11.3** |
| CausalHeteroGNN Avg-Group Acc | 78.2 | **70.8±4.5** |
| MLP content-only (bổ sung) | không có | 15.6±3.1 / 56.5±1.0 |

**Nguồn:** `results/final_tables.md` phần "Worst-group accuracy — conf"

### Bảng 1 — Kết quả chính
- Bổ sung chú thích `†` làm rõ: CausalHeteroGNN dùng CLIPcons + 5 seeds, các baseline 3 seeds không CLIPcons
- Các con số không thay đổi (đã đúng trước đó)

### Bảng 4 — LOCO (mới, có IRM/EERM)
Bảng LOCO trước chỉ có CausalHeteroGNN, Baseline GNN, MLP. Đã bổ sung IRM và EERM.

---

## 2. Thí nghiệm LOCO IRM/EERM — đã chạy thành công

Ba fold LOCO đã chạy bằng `pipeline/08_baselines_irm_eerm.py` (single seed=42):

| Fold | IRM unseen Acc% | EERM unseen Acc% |
|---|---|---|
| loco_a (nottheonion + pareidolia) | 65.6% | 62.2% |
| loco_b (upliftingnews + fakehistoryporn) | 61.2% | 70.0% |
| loco_c (usnews+usanews + fakealbumcovers) | 62.6% | 80.8% |
| **Mean±std** | **63.1±1.8%** | **71.0±7.6%** |

**File kết quả mới:**
- `results/baselines_irm_eerm_locoa_s42.json`
- `results/baselines_irm_eerm_locob_s42.json`
- `results/baselines_irm_eerm_lococ_s42.json`

**Nhận xét:**
- IRM ổn định hơn (std=1.8) nhưng thấp hơn CausalHeteroGNN (69.4%)
- EERM trung bình (71.0%) tương đương MLP (71.6%) nhưng dao động rất lớn (std=7.6)
- EERM vượt trội ở loco_c (80.8%) nhưng kém hơn ở loco_a (62.2%)
- CausalHeteroGNN (69.4±5.4) ổn định nhất trong nhóm các mô hình GNN

---

## 3. Hình đã tạo/cập nhật

| File | Mô tả | Trạng thái |
|---|---|---|
| `figures/fig1_scm_dag.png` | SCM DAG — đường đi cửa sau C1→Y | Giữ nguyên |
| `figures/fig2_architecture.png` | Kiến trúc pipeline Neo4j + CausalHeteroGNN | Giữ nguyên |
| `figures/fig3_confounding_shift.png` | So sánh Confounding-Shift tất cả mô hình | Giữ nguyên |
| `figures/fig4_ood_comparison.png` | Held-Out OOD vs Conf-Shift — **cập nhật số** | **Đã cập nhật** |
| `figures/fig5_confusion_matrices.png` | Confusion matrix Baseline vs Causal | Giữ nguyên |
| `figures/fig6_fastrp_ablation.png` | Tác động rò rỉ FastRP | Giữ nguyên |
| `figures/fig7_lfr.png` | Label-Flip Rate dưới can thiệp | Giữ nguyên |
| `figures/fig8_dashboard.png` | BI Dashboard KPI tổng hợp | Giữ nguyên |
| `figures/fig9_loco_comparison.png` | **MỚI** — So sánh 5 mô hình trên 3 fold LOCO | **Mới tạo** |

**Thay đổi trong `generate_figures.py`:**
- `fig4_ood_comparison()`: cập nhật số liệu
  - `conf_means` = [52.1, 54.1, 59.6, 79.9] (trước: [36.4, 54.1, 59.6, 74.2])
  - `conf_stds` = [7.8, 4.7, 1.1, 4.2] (trước: [8.1, 4.7, 1.1, 3.6])
  - `std_means` = [57.6, 56.9, 60.9, 59.6] (trước: [58.4, 57.1, 60.9, 56.9])
  - Annotation cập nhật: "+20.3 pp CausalHeteroGNN+CLIPcons"
- `fig9_loco_comparison()`: **thêm mới** — tất cả 5 mô hình, 3 fold + mean

---

## 4. Phần bài đã sửa trong VN.md

| Phần | Nội dung thay đổi |
|---|---|
| Bảng 1 (5.3) | Thêm chú thích `†` về CLIPcons và số seeds |
| Caption Hình 3, 4 | Cập nhật cho rõ ràng hơn |
| Bảng 3 (5.4) | Sửa worst-group CausalHeteroGNN; thêm MLP vào bảng |
| Đoạn text 5.4 | Cập nhật nhận xét phản ánh số liệu đúng |
| Bảng 4 (5.5) | Bổ sung cột IRM và EERM; thêm ghi chú single-seed |
| Đoạn text 5.5 | Viết lại với phân tích IRM/EERM LOCO; thêm Hình 9 |

---

## 5. Cập nhật lần 2 (2026-06-13) — đa seed IRM/EERM + DOCX

### Kết quả đa seed (3 seeds {42,1,2}) cho LOCO IRM/EERM

| Fold | IRM seed42 | IRM seed1 | IRM seed2 | IRM mean±std |
|---|---|---|---|---|
| loco_a | 65.6% | 56.0% | 64.4% | **62.0±4.3%** |
| loco_b | 61.2% | 67.4% | 70.8% | **66.5±4.0%** |
| loco_c | 62.6% | 70.4% | 74.2% | **69.1±4.8%** |
| Grand mean | — | — | — | **65.8±2.9%** |

| Fold | EERM seed42 | EERM seed1 | EERM seed2 | EERM mean±std |
|---|---|---|---|---|
| loco_a | 62.2% | 62.2% | 62.3% | **62.2±0.1%** |
| loco_b | 70.0% | 72.1% | 70.1% | **70.7±1.0%** |
| loco_c | 80.8% | 73.9% | 76.1% | **76.9±2.9%** |
| Grand mean | — | — | — | **70.0±6.0%** |

### Files kết quả seed 1 và 2
- `results/baselines_irm_eerm_locoa_s1.json` — IRM=55.96%, EERM=62.17%
- `results/baselines_irm_eerm_locob_s1.json` — IRM=67.37%, EERM=72.08%
- `results/baselines_irm_eerm_lococ_s1.json` — IRM=70.44%, EERM=73.90%
- `results/baselines_irm_eerm_locoa_s2.json` — IRM=64.36%, EERM=62.29%
- `results/baselines_irm_eerm_locob_s2.json` — IRM=70.78%, EERM=70.13%
- `results/baselines_irm_eerm_lococ_s2.json` — IRM=74.21%, EERM=76.10%

### Thay đổi trong VN.md
- Bảng 4 LOCO: per-fold IRM/EERM cập nhật sang mean±std 3 seeds
- Mean row: IRM 63.1±1.8 → **65.8±2.9**, EERM 71.0±7.6 → **70.0±6.0**
- Kết luận (para 7): worst-group 56.0% → **37.8±11.3%**, thêm LOCO multi-seed
- Chú thích bảng: phân biệt †single seed / ‡3 seeds

### Thay đổi trong (VN)_Causal Graph.docx
| Thay đổi | Chi tiết |
|---|---|
| Abstract | Cập nhật 79.9%, 37.8%, LOCO multi-seed |
| Bảng 1 caption | Thêm ghi chú CLIPcons/seeds |
| Table 0 row 5 | "CausalHeteroGNN" → "CausalHeteroGNN (CLIPcons)" |
| Kết luận | 74.2%→79.9%, Baseline 36.4%→52.1%, thêm worst-group + LOCO |
| Mới: Section 5.3 | Phân tích LOCO 3 seeds |
| Mới: Bảng 3 | LOCO 5 models × 3 folds + mean |
| Mới: Section 5.4 | Worst-Group Accuracy |
| Mới: Bảng 4 | Worst-group: MLP/Baseline/IRM/EERM/CausalHeteroGNN |
| Backup | `(VN)_Causal Graph_BACKUP2.docx` |

## 6. Còn lại / Đã hoàn thành

| Hạng mục | Trạng thái |
|---|---|
| LOCO IRM/EERM đa seed | **HOÀN THÀNH** (3 seeds x 3 folds) |
| Số liệu Temporal split | Đã có, không thay đổi |
| File DOCX cập nhật | **HOÀN THÀNH** — update_vn_docx.py |
| VN.md cập nhật | **HOÀN THÀNH** |
| Citation | Giữ nguyên |

---

## Lưu ý quan trọng về nhất quán số liệu

- **VN.md Bảng 1 dùng Confounding-Shift numbers** từ hai nguồn:
  - Baseline ERM/IRM/EERM/MLP: `final_tables.md` "Confounding-Shift" (3 seeds, không CLIPcons)
  - CausalHeteroGNN: `final_tables.md` "Improvement v2 — Conf (CLIPcons, 5 seeds)"
  - Đây là thiết kế có chủ đích: paper đề xuất mô hình đầy đủ với CLIPcons

- **Số 36.4% trong bi_dashboard_metrics_actual.json** là kết quả nhánh nội bộ (internal baseline branch của CausalHeteroGNN), KHÔNG phải Baseline GNN độc lập (52.1%). Không nhầm hai số này.

- **Tất cả LOCO là single-seed=42**, kể cả kết quả cũ. Ghi chú này đã được thêm vào Bảng 4.
