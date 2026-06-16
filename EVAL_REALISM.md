# EVAL_REALISM — Đánh giá trên các phép chia DỮ LIỆU THẬT (không synthetic)
### Trả lời phản biện "confounding-shift ρ=0.9→0.1 quá nhân tạo". Ngày: 2026-06-12.

> Không tải thêm dữ liệu — tái dùng 5,898 bài + embeddings có sẵn, chỉ chia lại
> và **tính lại toàn bộ thống kê node từ train của split mới** (chống leak).
> Sinh split: `11_make_alt_splits.py` → `data/processed_{temporal,loco_a,b,c}/`.
> Config đánh giá: v2 (CLIPcons), inductive content-only, cùng quy trình 03/04/07.

---

## Khung đánh giá 3 tầng (định vị lại vai trò từng protocol)

| Tầng | Protocol | Tính chất | Vai trò |
|---|---|---|---|
| 1. Tự nhiên | **LOCO** (4 folds: held-out gốc + 3 folds mới) + **Temporal** | dữ liệu thật 100% | Bằng chứng chính về tổng quát hóa |
| 2. Can thiệp | **Counterfactual do(·)** (đã có — Bảng 2 LFR) | giữ nguyên nội dung, chỉ đổi ngữ cảnh | Đo trực tiếp mức phụ thuộc confounder |
| 3. Stress-test | **Confounding-Shift** ρ=0.9→0.1 | có kiểm soát (kiểu ColoredMNIST) | Khuếch đại shortcut để so sánh cơ chế — KHÔNG phải "thế giới thật" |

---

## 1. LOCO — Leave-One-Community-Out (3 folds mới, seed 42, inductive)

Mỗi fold giữ-ra 1 cặp cộng đồng (toàn-Real + toàn-Fake → test ~50% fake, không
đoán được bằng phong cách cộng đồng). Protocol held-out gốc (neutralnews+theonion)
chính là fold thứ 4 cùng họ.

| Fold (held-out) | Causal+cc | ERM | MLP |
|---|---|---|---|
| nottheonion + pareidolia | 62.3 | 57.9 | 61.3 |
| upliftingnews + fakehistoryporn | 70.5 | 56.8 | 73.1 |
| usnews+usanews + fakealbumcovers | 75.5 | 54.7 | 80.5 |
| *(fold gốc: neutralnews + theonion, 5 seeds)* | *59.6* | *52.7* | *60.3* |
| **Mean 3 folds mới** | **69.4±5.4** (F1 0.677, AUC 0.755) | 56.5±1.3 (**F1 0.403**) | 71.6±7.9 (F1 0.706) |

**Phát hiện quan trọng nhất của cả đợt này:** trên dữ liệu **hoàn toàn tự nhiên**,
ERM (GNN thuần) sụp ở *mọi* fold — F1 chỉ 0.403 so với 0.677/0.706 — tức
**shortcut cộng đồng gây hại thật, không phải chỉ trong benchmark synthetic**.
Đây là bằng chứng tự nhiên trực tiếp cho luận điểm trung tâm của đề tài, thứ mà
protocol held-out 1-fold cũ (mọi model ~57-61) không đủ sức bộc lộ.

Causal+cc vs MLP: MLP nhỉnh hơn ở 2/3 folds (mean 71.6 vs 69.4, biến thiên giữa
folds lớn hơn: ±7.9 vs ±5.4) — nhất quán với kết luận "trần content" đã có:
trên cộng đồng hoàn toàn mới, đồ thị không cộng thêm; giá trị của causal là
**giữ được đồ thị mà không bị nó phản chủ** (ERM minh họa điều ngược lại).

## 2. Temporal split (train 70% cũ nhất → OOD = 10% mới nhất, 3 seeds)

Đặc điểm dữ liệu: 10% mới nhất có fake-rate chỉ 0.23 (lệch hẳn so với train 50/50)
→ temporal shift của Fakeddit chủ yếu là **label-prior shift**.

| Model | OOD Acc% | Macro-F1 | AUC |
|---|---|---|---|
| Causal+cc | 49.5±19.1 | 0.449±0.187 | 0.662±0.045 |
| ERM | 79.5±1.0 | 0.614±0.050 | 0.662±0.023 |
| MLP | 23.1±0.6 | 0.191±0.009 | **0.688±0.032** |

**Đọc đúng bảng này:** AUC của cả ba model gần như NHAU (0.66–0.69) — khả năng
*xếp hạng* tin giả không khác biệt; mọi chênh lệch accuracy/F1 đến từ **ngưỡng
quyết định bị lệch** khi prior đổi (ERM "ăn may" vì thiên về majority-Real;
MLP thiên Fake nên sụp acc dù AUC cao nhất; causal dao động mạnh ±19 giữa seeds).
→ Kết luận trung thực: **không model nào — kể cả mô hình đề xuất — xử lý được
prior shift**; đây là giới hạn chung và là future-work cụ thể (calibration /
threshold adaptation theo thời gian), không phải thất bại riêng của phương pháp.

## 3. Hai tầng còn lại (đã có sẵn, chỉ đổi vai trò trình bày)

- **Counterfactual do(·)**: giữ nguyên title/image, chỉ đổi subreddit/domain —
  LFR baseline 30.1% vs causal ≈0% (sanity check); do(D=credible) 15.5%.
  Chính là "Counterfactual Subreddit Swap" được đề xuất — không cần làm mới.
- **Confounding-Shift**: giữ nguyên số (Causal+cc 79.9 vs ERM 52.1...) nhưng
  trình bày là *controlled stress-test / diagnostic*, không phải "thế giới thật".

---

## Tác động lên paper (sửa tối thiểu)

1. §5.1 viết lại thành "ba lớp đánh giá": tự nhiên (LOCO 4 folds + temporal),
   can thiệp (do(·)), stress-test có kiểm soát (confounding-shift).
2. Thêm bảng LOCO (điểm nhấn: ERM F1 0.403 trên dữ liệu thật) + đoạn temporal
   với cách đọc AUC-vs-threshold ở trên.
3. Kết luận/limitation: thêm prior-shift là giới hạn chung + hướng calibration.

## Trạng thái sau 3 tầng — tổng kết một bảng

| Protocol | Causal+cc đứng đâu? |
|---|---|
| Confounding-shift (stress) | **Thắng áp đảo** (79.9 vs 59.6 EERM) |
| LOCO tự nhiên (4 folds) | **Thắng ERM rõ rệt** (69.4 vs 56.5), ≈ MLP (−2.2, trong dao động fold) |
| Temporal (prior shift) | Hòa về AUC; **không ai** xử lý được threshold — giới hạn chung |
| Counterfactual do(C₁) | Bất biến theo thiết kế (sanity ✓) |

→ Câu trả lời cho "phương pháp có ý nghĩa gì nếu chỉ tốt trên synthetic":
**không còn là 'chỉ tốt trên synthetic' nữa** — LOCO tự nhiên cho thấy GNN thuần
sụp vì shortcut thật và phương pháp cắt giữ được đồ thị an toàn; benchmark
synthetic chỉ là nơi cơ chế được *khuếch đại để đo đếm*, đúng vai trò của nó.

## Files
- Splits: `11_make_alt_splits.py` → `data/processed_temporal`, `data/processed_loco_{a,b,c}`
- Kết quả: `metrics_tmp_s{42,1,2}.json`, `metrics_loco{a,b,c}_s42.json`,
  `baselines_erm_mlp_{tmp_s*,locoa,locob,lococ}*.json`
