# Phản hồi 4 góp ý của reviewer (bài FAIR — CausalHeteroGNN / Fakeddit)

> Tài liệu này trả lời từng góp ý, kèm **cơ sở khoa học** và **đề xuất sửa/bổ sung cụ thể vào bài**. Số liệu lấy từ kết quả trung thực (content-only, đã loại rò rỉ FastRP, trung bình 3 seed {42,1,2}) trong `Fair_Article_VN/paper_full_VN.md`. Bản hội nghị rút gọn (file `.docx` EN) là bản reviewer đọc; một vài chỗ trong `.docx` còn dùng số cũ (MiniLM, 61.0/61.1) cần đồng bộ lại — xem mục 5 cuối tài liệu.

---

## Góp ý 1 — OOD thể hiện thế nào? Tại sao chuẩn bị dữ liệu theo OOD? Việc chuẩn bị dữ liệu đã đáp ứng OOD chưa?

### 1.1 Trả lời ngắn gọn
OOD (Out-Of-Distribution — dữ liệu lệch phân phối so với lúc huấn luyện) là **trọng tâm khoa học của bài**, không phải một thí nghiệm phụ. Lý do: nhãn trong Fakeddit được gán **ở cấp cộng đồng** — gần như mỗi subreddit toàn Thật hoặc toàn Giả (xem Hình 9c, scatter tỷ lệ giả ≈ 0 hoặc ≈ 1). Vì vậy "subreddit" trở thành **biến gây nhiễu (confounder) C₁** mở ra đường tắt `Subreddit → Nhãn`. Một GNN thường sẽ đạt độ chính xác rất cao **vì lý do sai** (đoán theo cộng đồng), và sụp đổ khi gặp cộng đồng mới. Bài toán thật sự **không phải** "tăng accuracy trên một phép tách tĩnh" mà là **"mô hình còn đúng không khi tín hiệu cộng đồng giả bị bỏ đi hoặc đảo ngược?"** → đó là lý do phải đánh giá theo OOD.

### 1.2 OOD được hiện thực hóa bằng HAI tập dữ liệu bổ trợ
Đây là điểm cần nhấn mạnh rõ trong bài, vì nó trả lời luôn câu "chuẩn bị dữ liệu đã đáp ứng OOD chưa":

| | (A) OOD giữ-cộng-đồng (Held-Out-Subreddit) | (B) OOD dịch-chuyển-gây-nhiễu (Confounding-Shift) |
|---|---|---|
| **Cách tạo** | Giữ trọn 2 subreddit `r/neutralnews` (toàn Thật) + `r/theonion` (toàn Giả) ra ngoài train/val, chỉ dùng để test | Thay subreddit thật bằng confounder tổng hợp `spur_fakebias`/`spur_realbias`; gán cộng đồng *báo hiệu* nhãn với xác suất ρ |
| **Tham số** | Train 5000 / Val 400 / Seen-test 200 / OOD-test 298 | Train/Seen ρ=0.9 (cộng đồng dự báo mạnh) → **OOD-test ρ=0.1 (tương quan ĐẢO NGƯỢC)** |
| **Mã nguồn** | `01_prepare_data.py` (`OOD_SUBREDDITS`) | `make_confounded_dataset.py` |
| **Mục đích** | Kiểm tra tổng quát hóa sang cộng đồng *chưa từng thấy* | Kiểm tra độ bền khi tương quan giả *thay đổi/đảo chiều* lúc triển khai |
| **Kết quả** | Mọi mô hình ngang nhau (~57–61%) | Phân kỳ mạnh: Baseline 36.4% vs Causal **74.2%** |

### 1.3 Vì sao cần TỚI 2 tập — và đây là điểm mạnh, không phải mâu thuẫn
- Giữ một cộng đồng ra ngoài là **cần nhưng chưa đủ**: khi node cộng đồng OOD chỉ mang đặc trưng mặc định (không thông tin), *không mô hình nào khai thác được nó*, nên cả baseline lẫn causal đều quay về dùng nội dung → ngang nhau (~57%). Điều này **đúng và phải báo cáo** (kết quả null trung thực): nó chứng minh mô hình của ta *không* tạo lợi ích giả ở nơi không có gì để bền.
- Một phép thử độ-bền-confounder *thực sự* cần một **dịch chuyển** của tín hiệu giả (có ích lúc train, gây hại lúc test). Benchmark Confounding-Shift (kiểu ColoredMNIST) tạo đúng điều đó, và nó **tách bạch rõ ràng** mô hình bền-confounder khỏi mô hình ăn-đường-tắt.
- ⇒ Thông điệp: *"Tách gỡ nhân quả có ích khi và chỉ khi có một tương quan giả lúc train bị dịch chuyển lúc triển khai."* Hai tập dữ liệu chính là hai mặt của luận điểm này.

### 1.4 Đánh giá tính phù hợp của phép chia dữ liệu (cho cả góp ý 4)
- ✅ Cân bằng nhãn 50/50 ở mọi tập (chống thiên lệch lớp).
- ✅ Đặc trưng hành vi (User/Subreddit/Domain) tính **chỉ từ train** → chống rò rỉ.
- ✅ Val chỉ dùng chọn siêu tham số + early-stopping; test giữ riêng cho đánh giá cuối.
- ✅ Seen-test lấy từ subreddit đã xuất hiện trong train; OOD-test lấy từ cộng đồng giữ-ra-ngoài → ranh giới sạch.
- ✅ Đã loại đặc trưng FastRP lúc test (nếu không sẽ thổi accuracy OOD từ ~57% lên ~94% — rò rỉ cấu trúc, xem §6.4).
- ⚠️ **Hạn chế cần ghi rõ:** tập OOD-test nhỏ (298 mẫu) → phương sai cao; đã giảm thiểu bằng trung bình 3 seed. *Khuyến nghị: tăng lên 5–10 seed* để báo cáo phương sai OOD trung thực hơn.

### 1.5 Cần sửa/bổ sung gì trong bài
1. Thêm một đoạn mở đầu Mục 5 (hoặc cuối Mục 3) định nghĩa rõ **OOD là gì trong ngữ cảnh bài này** (cộng đồng định nghĩa nhãn → confounder → đường tắt) trước khi nói cách chia dữ liệu.
2. Đặt bảng so sánh **(A) vs (B)** như §1.2 ở trên ngay đầu phần thực nghiệm để người đọc thấy "tại sao có 2 tập OOD".
3. Nêu rõ một câu kết luận về tính phù hợp của phép chia (mục 1.4) + hạn chế cỡ mẫu OOD + kế hoạch tăng seed.

---

## Góp ý 2 — Làm rõ luồng xử lý dữ liệu ở Hình 2 (từng khối, từng phương thức), và "tại sao C-1 chứ không phải C?"

### 2.1 Hình 2 hiện gồm 6 khối A→B→C-1→D→E→F — diễn giải từng bước

```
A. ĐẦU VÀO ĐA PHƯƠNG THỨC   →   B. DỰNG ĐỒ THỊ DỊ THỂ (Neo4j)   →   C₁. CAN THIỆP CỬA SAU CẤU TRÚC
        │                                                                       │
        └───────────────────────────────────────────────────────────┬─────────┘
                                                                      ▼
                                            D. HỌC HAI NHÁNH (Baseline ‖ Causal) + GRL
                                                                      │
                                                                      ▼
                                            E. HÀM MỤC TIÊU ĐA NHIỆM (7 thành phần)
                                                                      │
                                                                      ▼
                                            F. DỰ ĐOÁN BỀN VỚI OOD (2-way / 6-way / giải thích)
```

**Khối A — Đầu vào đa phương thức (mỗi loại xử lý ra sao, đầu ra là gì):**

| Loại dữ liệu | Bộ xử lý | Đầu ra (vector) | Vai trò SCM |
|---|---|---|---|
| **Post** (tiêu đề) | Sentence-Transformer `all-mpnet-base-v2` | **768-d** (chuẩn hóa L2) + 3 vô hướng (score, upvote_ratio, num_comments) → **771-d** | Nội dung X |
| **Image** | CLIP ViT-B/32 | **512-d** (chuẩn hóa L2) | Yếu tố nhân quả I |
| **User** | Thống kê hành vi (train) | **4-d**: post_count, avg_score, avg_upvote, fake_rate | Confounder C₂ |
| **Subreddit** | Thống kê cộng đồng (train) | **3-d**: post_count, fake_ratio, avg_score | **Confounder C₁** |
| **Domain** | Thống kê nguồn (train) | **3-d**: fake_ratio, avg_upvote, post_count | Yếu tố nhân quả D |

**Khối B — Dựng đồ thị dị thể & các đầu ra kết hợp ra sao:** 5 loại node trên được nối bằng 5 loại cạnh có hướng (`POSTED_BY`, `POSTED_IN`, `LINKS_TO`, `HAS_IMAGE`, `MEMBER_OF`), lưu trong Neo4j. **Cách kết hợp các phương thức:** *không* nối (concatenate) thô các vector khác chiều. Thay vào đó, mỗi loại node được **chiếu tuyến tính riêng** về cùng không gian ẩn `d=96`:
$$h_v^{(0)} = \text{Dropout}_{0.4}(\text{ReLU}(W_{\tau(v)} x_v + b_{\tau(v)}))$$
rồi **GraphSAGE 2 lớp** mới là nơi *trộn* đa phương thức: mỗi node Post gộp thông điệp từ Image/User/Subreddit/Domain lân cận (2-hop). Tức là **kết hợp diễn ra qua message-passing trên đồ thị, không phải nối vector** — đây là điểm cần nói rõ vì reviewer hỏi "các đầu ra khác nhau kết hợp ra sao".

**Khối C₁ — Can thiệp cửa sau cấu trúc (backdoor adjustment):** từ đồ thị gốc G, **xóa toàn bộ cạnh chạm tới node Subreddit**:
$$\mathcal{E}_{\text{causal}} = \{(u,v)\in E : \tau(u)\neq\text{Subreddit} \wedge \tau(v)\neq\text{Subreddit}\}$$
→ thu được đồ thị G_causal. Vì không có đường thông tin nào từ cộng đồng, biểu diễn nhân quả **chứng minh được là độc lập với C₁** → chặn đường cửa sau `X ← C₁ → Y`.

**Khối D — Học hai nhánh:**
- Nhánh **Baseline**: encode G đầy đủ → `h_base` (được phép thấy confounder).
- Nhánh **Causal**: encode G_causal (đã cắt Subreddit) → `h_causal`.
- **GRL (α=2.0)** + ràng buộc trực giao: ép `h_causal` không suy ra được subreddit, và `h_causal ⊥ h_base`.

**Khối E — Hàm mục tiêu đa nhiệm (7 thành phần):**
$$\mathcal{L} = \mathcal{L}_{base,2w} + 0.5\mathcal{L}_{base,6w} + \mathcal{L}_{causal,2w} + 0.5\mathcal{L}_{causal,6w} + 0.5\mathcal{L}_{spurious} + 0.5\mathcal{L}_{adv} + 0.2\mathcal{L}_{ortho}$$

**Khối F — Đầu ra:** phân loại 2-way (Thật/Giả), 6-way (True/Satire/Manipulated/Misleading/False/Partially-False), và giải thích nhân quả (LFR, đóng góp đường dẫn).

### 2.2 "Tại sao C-1 chứ không phải C?" — đây là LỖI nhãn không nhất quán, phải sửa
Trong chính Hình 2 đang có **3 cách gọi khác nhau cho cùng một thứ (subreddit)**:
- Khối A ghi: *"Subreddit — Confounder **C₁**"*
- Khối B ghi node types: *"Subreddit **(C)**"* ← chỗ này thiếu chỉ số dưới
- Nhãn khối ghi: *"**C-1** Structural Backdoor Adjustment"* ← gây hiểu "C-1" là một bước

→ Reviewer nhầm là dễ hiểu. **Khuyến nghị thống nhất ký hiệu trên toàn bài & hình:**
- Confounder cộng đồng = **C₁** (Subreddit); confounder hành vi = **C₂** (User). Luôn viết có chỉ số dưới.
- **Đổi tên nhãn khối** từ "C-1" thành một chữ cái thứ tự **"C"** (A, B, **C**, D, E, F) để không lẫn với confounder C₁. Tiêu đề khối nên là **"C. Can thiệp cửa sау cấu trúc (cô lập confounder C₁)"**.
- Trong khối B sửa "Subreddit (C)" → **"Subreddit (C₁)"**, và "User (U)" → **"User (C₂)"** cho khớp vai trò SCM.

### 2.3 Khuyến nghị vẽ thêm 1 hình ĐƠN GIẢN giải thích từng phương thức
Reviewer đề nghị "vẽ hình đơn giản để cắt nghĩa từng phần". Đề xuất một hình phụ (Hình 2b) chỉ tập trung **luồng từng phương thức → vector → hợp nhất qua đồ thị**:

```
Tiêu đề  ──mpnet──► [768] ─┐
score/upvote/#cmt ─► [3] ──┼─► Post x_P [771] ──┐
Ảnh ─────CLIP─────► [512] ─────► Image x_I [512]─┤
                                                 ├─►  W_τ (chiếu về 96-d mỗi loại)
User  ──thống kê──► [4]  ────► x_U [4] ──────────┤        │
Subreddit ────────► [3]  ────► x_S [3] ──────────┤        ▼
Domain ───────────► [3]  ────► x_D [3] ──────────┘   GraphSAGE×2 (TRỘN đa phương thức
                                                       qua message-passing 2-hop)
                                                            │
                                                            ▼
                                                       h_P [96]  → nhánh Causal / Baseline
```
Hình này nên kèm 1 câu: *"Các phương thức không được nối thô; chúng được chiếu về cùng 96-d rồi hợp nhất qua truyền thông điệp trên đồ thị."* (Có thể sinh bằng matplotlib — xem `recommend/` đính kèm nếu cần.)

---

## Góp ý 3 — Bảng kết quả: IRM/EERM có liên quan không? Tại sao chọn chúng? Chúng có sinh ra để giải OOD không? Số liệu thấp đi thì so sánh để làm gì, có công bằng không?

### 3.1 Trả lời ngắn gọn
**Có, rất liên quan, và so sánh là CÔNG BẰNG** — vì cả ba (IRM, EERM, CausalHeteroGNN) đều giải **đúng một bài toán**: học biểu diễn **bất biến/tổng quát hóa OOD** dưới dịch chuyển phân phối. IRM và EERM **được sinh ra chính xác để giải OOD** (đây là các phương pháp invariant learning kinh điển). Việc số liệu của chúng thấp hơn **không phải là điểm trừ của bài**, mà chính là **bằng chứng cho luận điểm**.

### 3.2 IRM và EERM là gì, có phải dành cho OOD không?
- **IRM (Invariant Risk Minimization, Arjovsky 2019)** — tìm biểu diễn sao cho **một bộ phân loại tối ưu đồng thời trên mọi môi trường**; mục tiêu thẳng là tổng quát hóa OOD. ✅ Sinh ra cho OOD.
- **EERM (Wu 2022, ICLR)** — *"Handling distribution shifts on graphs"*, chuyên cho **GNN/đồ thị**, sinh môi trường ảo và phạt phương sai rủi ro giữa môi trường. ✅ Sinh ra cho OOD **trên đồ thị** (cùng họ với bài này).

→ Đây không phải các mô hình "không liên quan" được kéo vào cho có. Chúng là **các baseline mạnh, đúng dòng (state-of-the-art invariant learning)**, và EERM còn cùng *miền đồ thị*.

### 3.3 Vì sao so sánh là công bằng (kiểm soát biến)
Tất cả dùng **CÙNG backbone HeteroGraphSAGE**: cùng phép chiếu, cùng 2 lớp SAGE, cùng d=96, cùng optimizer, cùng early-stopping, cùng seed, cùng giao thức đánh giá. **Khác biệt duy nhất là cơ chế khử confounder:**

| Phương pháp | Loại ràng buộc | Bản chất |
|---|---|---|
| Baseline | (không) | dùng tự do đường tắt |
| GRL đơn lẻ | mềm (đối kháng) | *làm nản* dùng confounder |
| IRM | mềm (phạt gradient) | *làm nản* |
| EERM | mềm (phạt phương sai rủi ro) | *làm nản* |
| **CausalHeteroGNN** | **cứng (cắt cạnh)** | **cấm** đường thông tin |

⇒ Vì chỉ khác *cơ chế*, chênh lệch kết quả quy được **đúng cho cơ chế**, không phải cho kiến trúc/dữ liệu → **đây chính là định nghĩa của so sánh công bằng (controlled comparison)**.

### 3.4 "Số liệu thấp đi thì so sánh để làm gì?" — đây là điểm cốt lõi cần giải thích lại trong bài
Reviewer hiểu nhầm rằng "số thấp = mô hình so sánh kém = so sánh vô nghĩa". Ngược lại:

1. **Trên tập dịch-chuyển-gây-nhiễu**, số *cao hơn = bền hơn*. Kết quả cho một **thứ bậc đơn điệu, ổn định qua 3 seed**:
   $$\text{Baseline } 36.4\% < \text{IRM } 54.1\% < \text{EERM } 59.6\% < \textbf{Causal } 74.2\%$$
   IRM/EERM **cao hơn baseline** (chúng *có* tác dụng khử confounder một phần) nhưng **thấp hơn structural cut** → chứng minh: *ràng buộc cứng (cắt cạnh) đáng tin hơn các hình phạt mềm dưới tương quan giả mạnh.* Đây là **đóng góp khoa học**, không phải khiếm khuyết.

2. **Vì sao mềm thua cứng:** khi tương quan giả lúc train rất mạnh (ρ=0.9), đường tắt "đáng giá" đến mức bộ tối ưu *trả bù* được phần phạt mềm và vẫn giữ lại phụ thuộc confounder còn sót → sụp khi đảo chiều. Cắt cạnh thì *không còn đường nào để đánh đổi*. (Bài đã lập luận ở §6.6.)

3. **Trên tập OOD tiêu chuẩn**, cả 4 ngang nhau (~57–61%) — và bài **chủ động báo cáo kết quả null này**. Điều đó làm so sánh *trung thực hơn*, vì nó cho thấy không ai (kể cả ta) tạo lợi ích giả ở nơi không có dịch chuyển để khai thác.

### 3.5 Lưu ý quan trọng về "tính công bằng" mà bài đã tự nêu (giữ lại trong phần Hạn chế)
- EERM bản gốc dùng edge-editer học bằng REINFORCE; ở đây hiện thực bằng **nhiễu cạnh ngẫu nhiên** (biến thể thực dụng) → có thể *đánh giá thấp* trần của EERM. Bài đã ghi rõ điều này (§7.5) → **giữ nguyên**, đó là sự trung thực, và dù EERM mạnh hơn cũng *không thể* đạt bất biến *chính xác* (LFR=0) như cắt cạnh.
- Khẳng định của bài là **thứ tự xếp hạng** giữa các cơ chế trên cùng benchmark, **không** phải con số tuyệt đối lúc triển khai.

### 3.6 Cần sửa/bổ sung gì trong bài
1. Thêm 1 câu ngay dưới Bảng kết quả (Table 1 docx / Bảng 2 full): *"IRM và EERM là các phương pháp học-bất-biến được thiết kế riêng cho tổng quát hóa OOD (EERM dành cho đồ thị); chúng được chọn vì cùng giải đúng bài toán này và cho phép so sánh có kiểm soát trên cùng backbone."*
2. Thêm câu giải nghĩa thứ bậc: *"Số OOD cao hơn nghĩa là bền hơn; IRM/EERM > Baseline xác nhận chúng có tác dụng, còn < Causal xác nhận ràng buộc cứng vượt hình phạt mềm."*
3. Đảm bảo phần Hạn chế giữ ghi chú về biến thể EERM (đã có ở `paper_full_VN.md §7.5`, nhưng bản `.docx` rút gọn **chưa có** — nên thêm 1 câu vào docx).

---

## Góp ý 4 — Giải thích rõ bản chất mô hình đề xuất: các bước, xử lý dữ liệu từng bước, phép chia dữ liệu đã phù hợp chưa?

### 4.1 Bản chất mô hình (1 câu)
CausalHeteroGNN = **HeteroGraphSAGE hai nhánh song song** trong đó **nhánh nhân quả mã hóa một đồ thị đã bị cắt bỏ toàn bộ cạnh Subreddit** (hiện thực hóa *backdoor adjustment* của Pearl ở cấp kiến trúc), cộng thêm GRL + trực giao làm điều chuẩn phụ → buộc mô hình phân loại bằng **nội dung** thay vì **đường tắt cộng đồng**.

### 4.2 Các bước xử lý (end-to-end) — gộp lại để đưa vào bài
1. **Tiền xử lý & trích đặc trưng:** tiêu đề → mpnet 768-d; ảnh → CLIP 512-d; +3 vô hướng tương tác; đặc trưng hành vi User/Subreddit/Domain tính **chỉ từ train**.
2. **Dựng HIN trên Neo4j:** 5 node × 5 cạnh; GDS chạy PageRank/Louvain/Betweenness/FastRP/NodeSimilarity (FastRP **chỉ dùng cho BI**, *loại khỏi input GNN lúc test* để tránh rò rỉ).
3. **Chiếu theo loại node** về d=96 (mỗi loại 1 ma trận W riêng) + ReLU + Dropout 0.4.
4. **GraphSAGE 2 lớp** trộn đa phương thức qua message-passing 2-hop → `h_P`.
5. **Hai nhánh:** Baseline (G đầy đủ → `h_base`) và Causal (G_causal đã cắt Subreddit → `h_causal`).
6. **Tách gỡ:** GRL(α=2.0) ép `h_causal` ⟂ thông tin subreddit; trực giao ép `h_causal ⊥ h_base`.
7. **Đa nhiệm:** 7 thành phần loss (2-way + 6-way cho mỗi nhánh + spurious + adv + ortho).
8. **Suy luận:** OOD tiêu chuẩn dùng giao thức **quy nạp không rò rỉ** (che cạnh test, bỏ FastRP); Confounding-Shift dùng **truyền dẫn** (vì subreddit chính là biến đang kiểm tra).
9. **Giải thích:** can thiệp `do(C₁=swap)`, `do(I=∅)`, `do(D=credible)` + đo **Label-Flip-Rate** + đóng góp đường dẫn theo gradient.

### 4.3 Phép chia dữ liệu đã phù hợp chưa?
**Phù hợp** — xem đánh giá chi tiết ở §1.4 (cân bằng nhãn, đặc trưng train-only chống rò rỉ, val/test tách bạch, ranh giới seen/OOD sạch, đã khử rò rỉ FastRP). **Hai điểm cần ghi rõ vào bài:** (i) cỡ OOD-test nhỏ (298) → khuyến nghị tăng số seed; (ii) trade-off 6-way: ép rời cộng đồng làm Macro-F1 6 lớp giảm (~0.20–0.30) vì một số lớp (vd Satire) gắn chặt với cộng đồng — đã nêu ở §7.3, nên giữ.

### 4.4 Cần sửa/bổ sung gì trong bài
1. Đưa **danh sách 9 bước §4.2** thành một hộp "Pipeline tóm tắt" đầu Mục 4 — trực tiếp trả lời "các bước như thế nào".
2. Với mỗi phương thức, ghi rõ **đầu vào → bộ xử lý → số chiều đầu ra** (bảng ở §2.1) trong Mục 4.
3. Nói rõ **cách hợp nhất là qua message-passing, không nối vector** (điểm hay bị bỏ sót).

---

## 5. Các điểm KHÔNG nhất quán giữa bản `.docx` (reviewer đọc) và bản full — nên đồng bộ trước khi nộp

| Hạng mục | Bản `.docx` EN | Bản `paper_full_VN.md` (đúng/mới) | Xử lý |
|---|---|---|---|
| Số node/quan hệ | "17,079 nodes / 28,274 rels" (mục 3) | 17,719 nodes / 27,974 rels | Thống nhất theo bản full (kiểm lại số thực từ Neo4j) |
| Acc OOD tiêu chuẩn (sau khi bỏ FastRP) | 61.0% / 61.1% (số MiniLM cũ) | 57.2% / 57.7% (mpnet, 3 seed) | Cập nhật docx về 57.2/57.7 |
| Ghi chú biến thể EERM (REINFORCE → nhiễu cạnh) | **thiếu** | có (§7.5) | Thêm 1 câu vào docx |
| Nhãn khối Hình 2 ("C-1", "Subreddit (C)") | không nhất quán | — | Sửa hình theo §2.2 |

> Ưu tiên: (1) sửa nhãn Hình 2 (góp ý 2 — dễ nhất, hiệu quả cao); (2) thêm 2 câu giải thích IRM/EERM dưới bảng (góp ý 3); (3) thêm hộp pipeline 9 bước + bảng phương thức (góp ý 4); (4) đồng bộ số liệu docx ↔ full (mục 5).
