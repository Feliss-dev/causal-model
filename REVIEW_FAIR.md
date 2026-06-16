# Hội đồng phản biện — Đánh giá toàn diện nghiên cứu CausalHeteroGNN (Fakeddit)

> Vai trò: FAIR/NeurIPS/ICML Reviewer + GS hướng dẫn Causal ML/GNN/OOD + Senior ML Engineer.
> Đầu vào: paper EN (docx) + 6 file source code + đối chiếu số liệu với `results/metrics_*.json` thực tế.
> Ngày đánh giá: 2026-06-11.

---

## 1. Executive Summary

**Điểm mạnh:**
- Pipeline kỹ thuật nghiêm túc và trung thực hơn mức trung bình: loại FastRP khỏi input mặc định vì leakage (`03_train_gnn.py:52-57`), feature thống kê tính từ train-only (`01_prepare_data.py:197-266`), CROSS_POST/Comment bị xóa vì gây leak, 3 seeds với mean±std.
- Phát hiện FastRP leakage (Section 5.4) là đóng góp **thực sự có giá trị thực tiễn** — 94% → 61% là một con số cảnh tỉnh, kiểm chứng được, và hiếm bài nào ở mảng GNN misinformation chỉ ra rõ như vậy.
- Confounding-Shift Benchmark kiểu ColoredMNIST trên HIN là một protocol đánh giá hợp lý, có baseline IRM/EERM cùng backbone.
- Số liệu chính trong paper **khớp với file kết quả thực tế** (đã verify Table 1 cột Conf-Shift với `metrics_bd_s{42,1,2}.json` và `main_results_dashboard_metrics.json` — khớp hoàn toàn). Đây không phải số bịa.

**Điểm yếu:**
- Kết quả chủ lực (74.2% vs 36.4%) **thắng một phần by construction**: confounder synthetic được tiêm duy nhất qua node Subreddit, và cơ chế của model là xóa đúng các edge đó.
- "Baseline GNN" trong paper **không phải model độc lập** — nó là nhánh baseline bên trong chính CausalHeteroGNN (Major Issue M1).
- Causal claim được over-state: nhánh "causal" vẫn dùng các feature suy ra từ label train (Domain `fake_ratio_real`, User `fake_rate`).
- Một cột số liệu trong Table 1 không khớp dữ liệu (Major Issue M4).

**Mức độ sẵn sàng công bố:** Với FAIR (hội nghị quốc gia), bài ở mức **borderline-accept nếu sửa các điểm dưới đây**; với NeurIPS/ICML thì sẽ bị reject vì novelty và quy mô thực nghiệm (1 dataset, 5,898 posts, 3 seeds).

---

## 2. Major Issues

### M1. "Baseline GNN" không phải baseline độc lập — mâu thuẫn paper vs code
- **Paper (§5.2):** "Baseline GNN (HeteroGraphSAGE on the original graph without noise handling)".
- **Code thực tế:** số "baseline" trong `metrics_bd_*.json` là output của `baseline_clf_2way` — nhánh baseline **bên trong cùng một CausalHeteroGNN**, dùng chung encoder `conv1/conv2` với nhánh causal (`03_train_gnn.py:207-233`). Encoder này nhận gradient từ cả loss causal, adversarial GRL và orthogonality. Nó không phải một HeteroGraphSAGE huấn luyện độc lập.
- **Mức độ ảnh hưởng: CAO.** Joint training có thể làm nhánh baseline tệ hơn → khuếch đại khoảng cách 36.4 vs 74.2.
- **Sửa tối thiểu:** Train ERM standalone (tái dùng `GNNClassifier` trong `06_baselines_irm_eerm.py`), 3 seeds, thay số "Baseline GNN" + sửa 1 câu §5.2.

### M2. Benchmark thắng by construction — cần thừa nhận và rào lại claim
Trong `make_confounded_dataset.py`, biến env synthetic đi vào graph **chỉ qua 3 kênh**: node feature Subreddit (`fake_ratio_real` = 0.9/0.1), edge `posted_in`, edge `member_of` — và `_cut_confounder_edges` (`03_train_gnn.py:188-192`) xóa đúng toàn bộ các kênh này. Nhánh causal **không thể** thấy confounder, nên việc nó robust khi đảo ρ là hệ quả logic, không phải phát hiện thực nghiệm. LFR do(C₁)≈0.0% là **tautology**.
- **Sửa tối thiểu (chỉ text):** Định vị lại đóng góp: *benchmark chứng minh "khi confounder đã được nhận diện, structural cut loại bỏ nó triệt để hơn các soft penalty (IRM/EERM)"* — so sánh hard-cut vs soft-penalty chính là kết quả có giá trị thật (74.2 > 59.6 > 54.1 > 36.4, ordering nhất quán cả 3 seeds). Chuyển LFR do(C₁)=0% thành sanity check.

### M3. Nhánh "causal" vẫn dùng feature suy ra từ label — claim "stable content features" không chính xác
- `Domain.fake_ratio_real` và `User.fake_rate` là **target encoding** P(fake | domain/user) tính từ label train (`01_prepare_data.py:212,233`). Backdoor cut chỉ xóa Subreddit; nhánh causal vẫn message-pass qua Domain và User.
- Đây là lý do conf-shift causal OOD đạt **74.2%** trong khi standard OOD content-only chỉ **57.7%**: eval conf-shift chạy transductive, OOD posts vẫn nối tới Domain node có lịch sử fake_ratio từ train. LFR do(D=credible)=15.5% xác nhận model lệ thuộc kênh này.
- Mâu thuẫn nội tại: Causal DAG trong Neo4j (`02_neo4j_import.py:396-417`) khai báo **cả User(C2) là confounder**, nhưng model chỉ xử lý C1.
- **Sửa tối thiểu:** (a) sửa text thừa nhận nhánh causal dùng content + source/user history; (b) ablation: chạy conf-shift với Domain `fake_ratio_real` thay bằng hằng 0.5, báo cáo phần đóng góp của kênh domain-history.

### M4. Cột "Held-Out OOD" trong Table 1 không khớp dữ liệu
Tính lại từ `metrics_main_s{42,1,2}.json`:

| Model | Paper | Dữ liệu thực (3-seed mean) |
|---|---|---|
| Baseline GNN | 58.4 | **57.2** (58.4 là riêng seed 42) |
| CausalHeteroGNN | 56.9 | **57.7** |
| IRM | 57.1 | 56.9 (xấp xỉ, chấp nhận được) |
| EERM | 60.9 | 60.9 ✓ |

Paper hiện để baseline > causal trên held-out, trong khi dữ liệu 3-seed nói ngược lại. **Sửa:** 58.4→57.2 và 56.9→57.7.

### M5. Hai protocol dùng hai chế độ inference khác nhau mà paper không nói
- Standard OOD: **inductive content-only** — `mask_post_edges` (`04_evaluate.py:37-64`) xóa *mọi* edge chạm test Post → tại test time GNN suy biến thành MLP trên feature của post.
- Confounding-shift: **transductive** (full message passing).
- `03_train_gnn.py:482-502` chỉ mask edge có *source* là test Post — codebase có **hai định nghĩa "strict inductive" khác nhau** giữa 03 và 04/06.
- **Sửa tối thiểu:** thêm 2–3 câu vào §5.1 nêu rõ chế độ inference của từng protocol và lý do.

---

## 3. Minor Issues

1. §3 paper liệt kê FastRP là Post feature nhưng mọi kết quả chính chạy `USE_FASTRP=0`. Sửa thành "FastRP chỉ dùng trong phân tích leakage §5.4 và BI dashboard".
2. Docstring model ghi "Post: 384 (text)... = 451" (`03_train_gnn.py:107`) — thực tế dùng mpnet 768-d → 771. Comment stale.
3. `01_prepare_data.py:53-77`: khối comment mô tả 4 OOD subreddits nhưng config thực chỉ 2.
4. Metadata CausalDAG trong Neo4j stale: `ood_subreddits: "nottheonion, pareidolia"`, `grl_alpha: 1.0` (thực 2.0).
5. Random-fallback embeddings nếu CLIP/SentenceTransformer lỗi (`01_prepare_data.py:192,428`) — nên fail hard.
6. Min-max normalization tính trên toàn bộ node kể cả test (`03_train_gnn.py:291-292`) — leak nhẹ về feature scaling.
7. LFR "do(C₁ = swap)": code thực hiện swap sang subreddit có fake_ratio **thấp nhất** → mô tả chính xác: "do(C₁ = neutral)".
8. Model selection trên val có ρ=0.9 (val cũng confounded) — nên ghi 1 câu thừa nhận (chuẩn ColoredMNIST).
9. Không có statistical test; với baseline ±8.1 nên thêm ghi chú khoảng tin cậy.
10. EERM là practical variant (random edge perturbation, không REINFORCE) — đã khai báo trung thực, thêm citation rõ.

---

## 4. OOD Assessment

**A. OOD có thật không?** Có, ở hai mức:
- *Held-out subreddit shift* (thật): seen-test acc ~76-78% vs OOD ~57-61% (content-only) → shift thực về topic/style.
- *Confounding shift* (synthetic, có kiểm soát): ρ 0.9→0.1; bằng chứng mạnh nhất là **AUC 0.314 < 0.5** của baseline — decision boundary bị đảo thật.

**B. Pipeline dữ liệu:** Split đúng là OOD setup thật, không phải IID trá hình. Label leakage xử lý tốt (train-only aggregates, default 0.5 cho node unseen, FastRP đã loại). Leak còn sót: target-encoding Domain/User (M3), min-max norm toàn cục (minor #6).

**C. Kết luận:** Setup đạt, trung thực hơn đa số bài cùng mảng. Vấn đề nằm ở **cách diễn giải** kết quả conf-shift (M2, M3), không phải ở leak.

---

## 5. Causal Assessment

**Mức độ thuyết phục causal claim: 4/10.**

- DAG khai báo bằng tay, không học từ dữ liệu — chấp nhận được ở mức motivation.
- *"Backdoor adjustment" là cách gọi quá tay*: backdoor adjustment đúng nghĩa là ∑_c P(Y|X,c)P(c). Cái paper làm là **graph surgery / confounder removal**. Nên đổi thành "structural intervention (confounder-edge removal)".
- GRL + orthogonality là DANN-style invariant representation learning, không phải causal identification.
- Cái cứu điểm causal: LFR do(I=∅) 4%→8% và do(D=credible) 6.2%→15.5% là intervention **không tautological** — nên nhấn mạnh hơn.
- Định vị bài là *confounder-robust GNN với protocol đánh giá can thiệp*, không phải causal discovery.

---

## 6. Code Review

| File | Đánh giá |
|---|---|
| `01_prepare_data.py` | Tốt. Train-only stats, balanced sampling, OOD tách sạch. Sửa: comment 4-vs-2 subreddits, random fallback. |
| `02_neo4j_import.py` | Đạt. Import batch, constraints, GDS + NetworkX fallback. Sửa: CausalDAG metadata stale. GDS chạy trên full graph — OK vì chỉ phục vụ BI. |
| `03_train_gnn.py` | Kiến trúc khớp paper (loss weights đúng công thức §4). Early stopping theo val loss, gradient clipping, class weights — bài bản. Vấn đề: baseline branch dùng làm "Baseline GNN" (M1); mask 1 chiều khác 04/06 (M5); docstring dims stale. |
| `04_evaluate.py` | Temperature calibration trên val là điểm cộng. Mask 2 chiều nhất quán với 06. Counterfactual engine OK. |
| `05_dashboard.py` | Phục vụ BI/grading tốt, không ảnh hưởng claims khoa học. |
| `06_baselines_irm_eerm.py` | Viết tốt nhất repo: read-only, cùng backbone, cùng split. IRMv1 đúng chuẩn Arjovsky. EERM variant ghi chú trung thực. Thiếu: ERM standalone. |

---

## 7. Figure & Methodology Revision (Hình 2)

Reviewer sẽ thắc mắc: (i) feature dimension từng node type; (ii) encoder share hay riêng (code: **share**); (iii) GRL gắn vào đâu (code: vào h_c trước confounder classifier); (iv) orthogonality đo giữa h_c vs h_s; (v) tại sao nhánh baseline cũng có classifier riêng.

Sơ đồ đề xuất (khớp 100% code):

```
INPUT FEATURES                          SHARED ENCODER (weights dùng chung)
Post  = mpnet(768) ⊕ [score,upvote,    ┌──────────────────────────────┐
        n_comments](3) = 771-d   ──►   │ Linear proj per-type → 96-d  │
Image = CLIP ViT-B/32 (512-d)    ──►   │ ReLU + Dropout 0.4           │
User  = 4 stats (train-only)     ──►   │ HeteroSAGE ×2 (aggr=sum)     │
Subreddit = 3 stats              ──►   └──────┬───────────────┬───────┘
Domain    = 3 stats                           │               │
                                     G (full graph)   G_causal = G \ {edges chạm Subreddit}
                                              │               │
                                        h_post (96)     h_post_causal (96)
                                              │               │
                                      spurious_head     causal_head
                                              │ h_s           │ h_c
            L_ortho = |cos(h_c, h_s)| ◄───────┼───────────────┤
                                              │               ├─► GRL(α=2.0) ─► confounder_clf ─► L_adv
        confounder_clf ─► L_spurious ◄────────┘               │
                                              │               │
                                   clf 2-way + 6-way   clf 2-way + 6-way
                                       L_base              L_causal  ◄── prediction chính
```

Kèm bảng "Node/Edge schema": 5 node types (số lượng + dims) và 5 edge types + reverse (ToUndirected).

---

## 8. Baseline Review

- **IRM — hợp lý.** Env = phân hoạch theo subreddit chính là chiều confounder. Implementation đúng IRMv1.
- **EERM — hợp lý về lựa chọn**, implementation là biến thể đơn giản hóa — đã khai báo trung thực.
- **Công bằng:** backbone/hidden/layers/LR/early-stopping công bằng. Chưa công bằng: (1) "Baseline GNN" không độc lập (M1); (2) CausalHeteroGNN train multi-task còn IRM/EERM chỉ 2-way.
- **Baseline nên thêm:** (1) ERM standalone — bắt buộc; (2) MLP content-only — trả lời "graph đóng góp gì?"; (3) GroupDRO — ~40 dòng trên env partition có sẵn; (4) reweighting theo env.

---

## 9. Metrics Recommendation

**Giữ:** OOD Accuracy, OOD Macro-F1, OOD AUC (AUC 0.314 là bằng chứng đắt giá nhất), Relative F1-Drop, Seen Acc, mean±std 3 seeds.
**Thêm:** Worst-group accuracy (per-OOD-subreddit và per-env); ECE calibration (code temperature scaling đã có).
**Bỏ/hạ cấp:** LFR do(C₁) cho nhánh causal → sanity check; 6-way metrics nếu không bàn luận.
**Metric reviewer FAIR quan tâm nhất:** Conf-Shift OOD Acc + AUC + ordering nhất quán Baseline < IRM < EERM < Causal.

---

## 10. FAIR Acceptance Estimation

- **Hiện trạng:** Accept ~40% | Weak Reject ~35% | Reject ~25%.
- **Sau Minimal Revision Plan:** Accept ~70-75% | Weak Reject ~20% | Reject ~5-10%.

Lý do reject tiềm năng theo trọng số: by-construction benchmark + baseline không độc lập > causal claim quá tay > 1 dataset nhỏ, 2 OOD subreddits > số liệu Table 1 lệch.

---

## 11. Minimal Revision Plan

| # | Việc | Loại | Công sức |
|---|---|---|---|
| 1 | Train **ERM standalone** 3 seeds → thay số "Baseline GNN" + sửa §5.2 | Code + số | ~0.5 ngày |
| 2 | **Sửa Table 1 cột Held-Out OOD**: 58.4→57.2, 56.9→57.7 | Text | 10 phút |
| 3 | **Reframe causal claim**: "backdoor adjustment" → "structural intervention"; claim trung tâm = *hard cut > soft penalties*; LFR do(C₁)=0% → sanity check | Text | 0.5 ngày |
| 4 | **Ablation Domain-feature**: conf-shift với domain fake_ratio = 0.5, 3 seeds | Code + số | ~0.5 ngày |
| 5 | §5.1: nêu rõ chế độ inference khác nhau giữa 2 protocol | Text | 30 phút |
| 6 | §3: FastRP không nằm trong input GNN mặc định | Text | 10 phút |
| 7 | **MLP content-only baseline**, 3 seeds | Code + số | ~0.5 ngày |
| 8 | **Worst-group accuracy** per-subreddit/per-env | Code nhẹ | 1–2 giờ |
| 9 | Vẽ lại Fig 2 (dims, shared encoder, vị trí GRL/ortho) | Hình | 2–3 giờ |
| 10 | Code hygiene: CausalDAG metadata, docstring 384→768, comment 4-subreddit, random-fallback → hard fail | Code | 1–2 giờ |

Ba việc đầu là **bắt buộc**. Việc 4 và 7 là hàng rào phòng thủ tốt nhất. Không việc nào đụng đến pipeline dữ liệu, kiến trúc model, hay kết quả chủ lực 74.2%.
