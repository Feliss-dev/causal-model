# IMPROVEMENTS — Nâng cấp CausalHeteroGNN (v2)
### Kết quả thực nghiệm đầy đủ, kèm cả kết quả âm tính. Ngày: 2026-06-12.

> Mục tiêu đề ra (sau review): xóa 3 điểm yếu đã định lượng — (1) thua MLP trên
> standard OOD + satire worst-group thấp nhất bảng, (2) chỉ cắt được confounder
> *biết trước*, (3) thống kê mỏng. KHÔNG hứa "vượt trội mọi trường hợp"
> (no-free-lunch). Mọi số tái lập được; nguồn: `results/final_tables.md`.

---

## TÓM TẮT — trước vs sau

| Chỉ số | Trước (v1) | Sau (v2 = +CLIPcons) | Ghi chú |
|---|---|---|---|
| Conf-Shift OOD Acc | 74.2±3.6 | **79.9±4.2** (5 seeds) | AUC 0.851→**0.922**, F1-drop 12.7→**5.4%** |
| Conf-Shift Worst-Group | 37.8±11.3 | **56.0** | gấp ~2.4× ERM (21.0), 3.5× MLP (16.0) |
| Standard OOD Acc | 57.7±0.8 (thua MLP 60.5) | **59.6±1.9** ≈ MLP+cc 60.3±0.4 | **hòa trần content** (chênh 0.7 < std) |
| Standard Worst-Group (satire) | 29.6 (tệ nhất bảng) | **42.9** (cao nhất trong các model chính) | nhóm theonion\|Fake: 0.30→0.43 |
| Giả định confounder biết trước | bắt buộc | **xóa bỏ** — AutoCut v2 tự khám phá 3/3 seeds | xem mục 2 |
| Seeds cho dòng chủ lực | 3 | **5** {42,1,2,3,4} | |

---

## 1. CLIPcons — feature nhất quán tiêu đề–ảnh ✅ THÀNH CÔNG

**Cơ chế:** `compute_clip_consistency.py` → cosine(CLIP-text(title), CLIP-image)
— 1 scalar/post, append vào Post features (771→772). Mọi model nhận như nhau
(qua `build_heterodata`, flag `GNN_USE_CLIPCONS=1`) → so sánh công bằng.
Tín hiệu thật: cos(Real)=0.295 vs cos(Fake)=0.245 — tin giả lệch text-ảnh hơn
(đúng giả thuyết SAFE [12]).

**Kết quả (5 seeds):**
- Conf-shift Causal: 74.2 → **79.9±4.2** (+5.7đ), AUC 0.922, F1-drop chỉ 5.4%.
- Standard Causal: 57.7 → **59.6±1.9** — xóa khoảng cách với MLP (60.3±0.4).
- Satire (theonion|Fake): 30% → **43%**; worst-group stdood 29.6 → **42.9**
  (vượt MLP 34.9 — mô hình causal giờ là model TỐT NHẤT ở nhóm khó nhất).
- Điểm thú vị: MLP/ERM **không** hưởng lợi từ feature này (MLP 60.5→60.3;
  ERM còn bất ổn hơn) — lợi ích đến từ tương tác giữa tín hiệu consistency và
  message-passing của nhánh causal, không phải feature "tự mạnh".

**Files:** `metrics_cc{main,bd}_s{42,1,2,3,4}.json`,
`baselines_erm_mlp_cc{std,conf}_s*.json`, `worst_group_{stdood,conf}_cc.json`.

## 2. AutoCut v2 — TỰ khám phá confounder ✅ THÀNH CÔNG (qua 1 lần thất bại)

**v1 (gradient gating) — THẤT BẠI, báo cáo trung thực:** cổng học được trên từng
loại quan hệ + áp lực GRL adversarial → gates đứng nguyên ~0.93–0.97 kể cả khi
W_ADV=2.0 (encoder triệu-tham-số "giấu" env-info dễ hơn là đóng 10 gate vô hướng;
W_ADV cao còn phá nhánh causal — Seen F1 sụp còn 0.34). Code giữ lại sau flag
`GNN_AUTOCUT=1` làm bằng chứng âm tính. Nhất quán với phát hiện cũ "GRL đơn lẻ
không đủ disentangle".

**v2 (structural search + env-probe) — THÀNH CÔNG:** `10_autocut_search.py` duyệt
7 ứng viên cắt quan hệ; tiêu chí chọn = **env-probe accuracy trên validation**
(linear probe h_c → subreddit-ID; scale-free, không nhìn OOD). Lưu ý phương pháp:
tiêu chí IRM-penalty thô bị loại vì không bất biến theo thang loss (chọn sai
`posted_in` đơn lẻ, bỏ sót rò rỉ 2-hop qua `member_of`).

| Protocol | s42 | s1 | s2 | Kết luận |
|---|---|---|---|---|
| Confounded | posted_in+member_of (0.777) | posted_in+member_of (0.755) | posted_in+member_of (0.770) | **3/3 khám phá đúng phép cô lập Subreddit** — không được chỉ định, không nhìn OOD |
| Standard | member_of (0.410) | none (0.370) | has_image (0.445) | Bất ổn định, probe-gap nhỏ → **không có confounder trội** (đúng thực tế: không cut nào giúp OOD, 53–65% nhiễu) |

Probe của cut đúng (0.755–0.777) tách biệt rõ khỏi mọi cut sai (0.93–1.00) trên
dữ liệu confounded — biên phân biệt lớn, không phải may mắn. OOD của cut được
chọn: 76.8/82.9/86.6 ≈ hard-cut.

**Ý nghĩa:** hạn chế "phải biết trước confounder" của paper được thay bằng quy
trình 2 bước: *search-then-cut* với tiêu chí chọn thuộc phân phối huấn luyện
(DomainBed-style). Trên dữ liệu không có confounder trội, quy trình "từ chối
khám phá" — hành vi đúng, tránh cắt bừa.

## 3. GroupDRO ❌ KHÔNG GIÚP — loại

Thử 1 seed (stdood s42): OOD 55.7 vs 56.7 của causal gốc cùng seed — không cải
thiện, đúng dự đoán rằng DRO trên nhóm train không chuyển thành tổng quát hóa
sang cộng đồng chưa thấy. Code giữ sau flag `GNN_GROUPDRO=1`; không đưa vào bảng.

## 4. Củng cố thống kê ✅ MỘT PHẦN

- ✅ 5 seeds {42,1,2,3,4} cho cấu hình chủ lực (Causal+CLIPcons, MLP+cc, ERM+cc)
  trên cả 2 protocol; AutoCut search 3 seeds × 2 protocol.
- ⏸ Mở rộng OOD 4 subreddits (stretch E6): CHƯA chạy — cần tải lại ảnh
  (nottheonion, misleadingthumbnails — URL Reddit cũ tỷ lệ chết cao) và rebuild
  toàn bộ. `01_prepare_data.py` không chặn việc này; khuyến nghị làm trước khi
  nộp bài mở rộng. Đây là việc còn nợ duy nhất của kế hoạch.

---

## ĐÁNH GIÁ TỔNG THỂ SAU NÂNG CẤP (trung thực)

**Đã đạt:**
1. Trên conf-shift: bỏ xa mọi baseline (79.9 vs EERM 59.6) với worst-group 56.0
   và F1-drop 5.4% — claim "hard cut > soft penalty" mạnh hơn trước.
2. Trên standard OOD: từ "thua MLP" thành "hòa trần content về accuracy, dẫn đầu
   về worst-group/satire" — điểm yếu chí mạng cũ đã đóng.
3. Giả định confounder-biết-trước — phản biện nặng nhất về novelty — đã được
   thay bằng cơ chế khám phá có kiểm chứng âm/dương tính.

**Vẫn chưa (và nên nói thẳng trong bài):**
- "Vượt trội mọi trường hợp" là bất khả thi: khi không có shift, ERM dùng
  shortcut vẫn cao hơn trên seen (91% vs 84%). Đánh đổi này là bản chất.
- Satire 43% vẫn dưới 50% — cải thiện lớn nhưng chưa giải quyết; cần mô hình
  ngôn ngữ mạnh hơn cho nhiệm vụ này.
- Ablation domain-history (58.8) chưa chạy lại với CLIPcons — con số 79.9 vẫn
  bao gồm kênh domain (~15đ theo ablation v1); cần nêu rõ khi báo cáo.
- 1 dataset; OOD set 298 bài.

**Khuyến nghị sử dụng kết quả:** đủ chất lượng để (a) đưa vào bản camera-ready
như mục "Improvements & Confounder Discovery", hoặc (b) làm lõi cho bài tiếp theo
(AutoCut search là đóng góp mới đáng kể nhất). Chưa tự ý sửa docx — chờ quyết
định của tác giả.

## Tái lập

```powershell
# CLIPcons feature
uv run python compute_clip_consistency.py
# Causal+CLIPcons (vd conf, seed 42)
$env:GNN_USE_CLIPCONS="1"; $env:GNN_INPUT_DIR="data/processed_confounded"
$env:GNN_OOD_TRANSDUCTIVE="1"; $env:GNN_SEED="42"; $env:GNN_RUN_TAG="_ccbd_s42"
uv run python 03_train_gnn.py; uv run python 04_evaluate.py
# AutoCut v2 search
uv run python 10_autocut_search.py
# Tổng hợp
uv run python 09_final_tables.py
```
