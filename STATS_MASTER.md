# STATS MASTER — Toàn bộ số liệu của đề tài (cập nhật 2026-06-12)

> Tổng hợp duy nhất, đầy đủ. Mỗi bảng ghi rõ file nguồn trong `results/`.
> Quy ước: mean±std; v1 = sau revision (3 seeds); v2 = +CLIPcons (5 seeds).

---

## 0. DỮ LIỆU

| Hạng mục | Giá trị |
|---|---|
| Tổng bài (có ảnh thật) | 5,898 posts — Fakeddit 2008–2020 |
| Đồ thị | 17,079 nút (5,898 Post; 4,604 User; 658 Domain; 21 Subreddit; 5,898 Image), 28,274 quan hệ ×5 loại |
| Split gốc | train 5,000 (50/50) / val 400 / test 498 (200 seen + 298 OOD: neutralnews+theonion) |
| Subreddit train | 19 cộng đồng — **tất cả thuần nhãn** (fake_rate = 0.0 hoặc 1.0) → confounding cực đoan tự nhiên |
| Feature Post | v1: 771-d (mpnet 768 + 3 scalar); v2: 772-d (+clip_cons) |
| Tín hiệu clip_cons | cos(Real)=0.295 vs cos(Fake)=**0.245** — tin giả lệch text-ảnh hơn |
| Conf-shift benchmark | ρ=0.9 (train/val/seen) → 0.1 (OOD), 2 env synthetic |
| Temporal split | train 70% cũ / OOD = 10% mới nhất (590 bài, fake-rate 0.23 — prior shift) |
| LOCO folds | A: nottheonion+pareidolia (822, fr .46); B: upliftingnews+fakehistoryporn (616, .44); C: usnews+usanews+fakealbumcovers (318, .47) |

---

## 1. BẢNG CHÍNH v1 — 2 protocol (3 seeds) `[final_tables.md]`

### Confounding-Shift (transductive)
| Model | Seen% | OOD% | OOD F1 | AUC | F1-Drop% | Worst-Grp% |
|---|---|---|---|---|---|---|
| BaselineBranch (diagnostic) | 90.8±0.6 | 36.4±8.1 | 0.333 | 0.314 | 63.3 | 7.6 |
| ERM (baseline độc lập) | 91.0±0.4 | 52.1±7.8 | 0.499 | 0.511 | 45.1 | 23.5 |
| IRM | 91.2±0.2 | 54.1±4.7 | 0.517 | 0.524 | 43.2 | 24.4 |
| MLP (content-only) | 81.7±0.6 | 59.5±1.7 | 0.571 | 0.685 | 30.0 | 15.6 |
| EERM | 93.2±0.2 | 59.6±1.1 | 0.572 | 0.694 | 38.6 | 28.6 |
| **CausalHeteroGNN** | 83.8±0.5 | **74.2±3.6** | **0.731** | **0.851** | **12.7** | **37.8** |

### Standard OOD — held-out neutralnews+theonion (inductive)
| Model | Seen% | OOD% | OOD F1 | AUC | Worst-Grp% |
|---|---|---|---|---|---|
| IRM | 73.0 | 56.9±3.0 | 0.511 | 0.614 | 28.9 |
| ERM | 75.3 | 57.6±2.7 | 0.541 | 0.627 | 30.4 |
| CausalHeteroGNN | 76.5 | 57.7±0.8 | 0.538 | 0.596 | 29.6 |
| MLP | 81.8 | 60.5±1.1 | 0.582 | 0.687 | 37.1 |
| EERM | 75.0 | 60.9±1.9 | 0.566 | 0.653 | 33.3 |

---

## 2. NÂNG CẤP v2 — CLIPcons (5 seeds) `[metrics_cc*, worst_group_*_cc]`

| Chỉ số | v1 | **v2** | Δ |
|---|---|---|---|
| Conf OOD Acc | 74.2±3.6 | **79.9±4.2** | +5.7 |
| Conf AUC | 0.851 | **0.922** | +0.071 |
| Conf F1-Drop | 12.7% | **5.4%** | −7.3 |
| Conf Worst-Group | 37.8 | **56.0** | +18.2 |
| Std OOD Acc | 57.7±0.8 | **59.6±1.9** (MLP+cc 60.3±0.4) | hòa trần content |
| Std Worst-Group | 29.6 (tệ nhất) | **42.9** (tốt nhất; MLP 34.9) | +13.3 |
| Satire theonion\|Fake | 30% | **43%** | +13 |

MLP/ERM không hưởng lợi từ CLIPcons (MLP 60.5→60.3; ERM+cc conf 55.0±4.0, std 52.7±1.1
— bất ổn) → lợi ích đặc thù của nhánh causal.

---

## 3. AUTOCUT v2 — tự khám phá confounder `[autocut_search_*]`

Tiêu chí: argmin env-probe accuracy trên validation (KHÔNG nhìn OOD).

| Protocol | s42 | s1 | s2 |
|---|---|---|---|
| Confounded | **posted_in+member_of** (0.777; OOD 76.8) | **posted_in+member_of** (0.755; 82.9) | **posted_in+member_of** (0.770; 86.6) |
| Standard | member_of (0.410) | none (0.370) | has_image (0.445) |

- Confounded: **3/3 đúng** phép cô lập Subreddit; probe của cut đúng (0.755–0.777)
  tách biệt rõ mọi cut sai (0.93–1.00).
- Standard: bất ổn định, gap nhỏ → "không có confounder trội" — hành vi đúng.
- Kết quả âm tính (giữ minh bạch): gradient-gating thất bại (gates kẹt ~0.93;
  W_ADV=2.0 phá nhánh causal — Seen F1 0.34); IRM-penalty làm tiêu chí chọn SAI
  (chọn posted_in đơn lẻ, sót rò 2-hop member_of).

---

## 4. ABLATION & CHẨN ĐOÁN

| Thí nghiệm | Kết quả | File |
|---|---|---|
| Domain-history neutral (=0.5) | Causal conf 74.2 → **58.8±2.9** (AUC 0.851→0.681) → ~15.4đ từ lịch sử nguồn; phần còn lại = trần content | `metrics_nd_s*` |
| FastRP leakage | OOD ảo 93.6/94.3% → **61.0/61.1%** khi loại (lệch +33đ) | `metrics_s*` vs `metrics_nofrp_s*` |
| GroupDRO | 55.7 vs 56.7 cùng seed — không giúp, loại | `metrics_gdro_s42` |
| LFR do(C₁=neutral) | Baseline 30.1% vs Causal **≈0.0%** (sanity check by construction) | Bảng 2 paper |
| LFR do(I=∅) / do(D=credible) | 4.0→8.0% / 6.2→**15.5%** — causal dựa nhiều hơn vào content+source | Bảng 2 paper |

---

## 5. ĐÁNH GIÁ DỮ LIỆU THẬT (realism) `[EVAL_REALISM.md]`

### LOCO — 3 folds tự nhiên mới (seed 42, inductive, config v2)
| Fold held-out | Causal+cc | ERM | MLP |
|---|---|---|---|
| nottheonion+pareidolia | 62.3 | 57.9 | 61.3 |
| upliftingnews+fakehistoryporn | 70.5 | 56.8 | 73.1 |
| usnews+usanews+fakealbumcovers | 75.5 | 54.7 | 80.5 |
| **Mean** | **69.4±5.4** (F1 0.677, AUC 0.755) | 56.5±1.3 (**F1 0.403**) | 71.6±7.9 (F1 0.706) |

→ **GNN thuần sụp vì shortcut trên dữ liệu THẬT** (F1 0.403 mọi fold) — bằng chứng
tự nhiên trực tiếp cho luận điểm đề tài. Causal ≈ MLP (−2.2đ, trong dao động fold).

### Temporal (3 seeds; OOD = 10% mới nhất, fake-rate 0.23)
| Model | Acc% | Macro-F1 | AUC |
|---|---|---|---|
| Causal+cc | 49.5±19.1 | 0.449 | 0.662 |
| ERM | 79.5±1.0 | 0.614 | 0.662 |
| MLP | 23.1±0.6 | 0.191 | **0.688** |

→ AUC ba model ≈ nhau: temporal shift của Fakeddit = **prior shift**; chênh lệch
acc/F1 là artifact ngưỡng. Không model nào xử lý được — giới hạn chung (future
work: calibration theo thời gian). Không trích accuracy temporal mà thiếu caveat này.

---

## 6. BỨC TRANH CUỐI — vị thế CausalHeteroGNN (v2) qua 4 loại kiểm tra

| Kiểm tra | Vị thế |
|---|---|
| Confounding-shift (stress-test) | **Thắng áp đảo**: 79.9 vs EERM 59.6, worst-group 56.0 |
| LOCO tự nhiên (4 folds) | **Thắng ERM rõ** (69.4 vs 56.5), ≈ MLP |
| Standard held-out | Hòa trần content (59.6 vs 60.3), **nhất worst-group/satire** |
| Temporal (prior shift) | Hòa AUC; không ai vượt — giới hạn chung |
| Counterfactual do(C₁) | Bất biến theo thiết kế ✓ |

## 7. FILE MAP

| Nhóm | Files |
|---|---|
| Bảng gộp | `results/final_tables.{md,json}` (sinh bởi `09_final_tables.py`) |
| v1 chính | `metrics_{main,bd}_s*`, `baselines_irm_eerm_*`, `baselines_erm_mlp_{stdood,conf}_s*` |
| v2 | `metrics_cc{main,bd}_s{42,1,2,3,4}`, `baselines_erm_mlp_cc{std,conf}_s*`, `worst_group_*_cc.json` |
| AutoCut | `autocut_search_{conf,std}_s{42,1,2}.json` |
| Ablation | `metrics_nd_s*`, `metrics_nofrp_s*`, `metrics_s*` (FastRP leaky), `metrics_gdro_s42` |
| Realism | `metrics_tmp_s*`, `metrics_loco{a,b,c}_s42`, `baselines_erm_mlp_{tmp,loco*}*` |
| Báo cáo | `REVIEW_FAIR.md`, `RESULTS_UPDATE.md`, `IMPROVEMENTS.md`, `EVAL_REALISM.md`, `RUNNING_GUIDE.md`, `RESPONSE_TO_COMMITTEE.md` |
