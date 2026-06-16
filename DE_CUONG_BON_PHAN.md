# BỐN THÀNH PHẦN NGHIÊN CỨU — BẢN CHI TIẾT
### Đề tài: Phân rã quan hệ nhân quả trên đồ thị không đồng nhất bằng GraphSAGE cho phát hiện tin giả đa phương thức

> Bản đầy đủ nhất, thống nhất với `Paper_VN_full.md` (bản viết lại) và
> `STATS_MASTER.md` (số liệu). Mỗi mục có thể rút gọn tùy biểu mẫu;
> phần *in nghiêng* là căn cứ/giải trình kèm theo khi hội đồng hỏi.

---

## 1. MỤC TIÊU NGHIÊN CỨU

**Mục tiêu tổng quát:**
Xây dựng một phương pháp phát hiện tin giả đa phương thức trên mạng thông tin
không đồng nhất, có khả năng **bền vững trước các biến gây nhiễu cấu trúc trong
đồ thị**, đặc biệt là sự phụ thuộc giả tạo giữa nhãn tin thật/tin giả và lịch sử
phân bố nhãn theo cộng đồng hoặc nguồn đăng. Thay vì chỉ tối ưu độ chính xác
trên phân phối đã thấy, nghiên cứu hướng tới việc nhận diện, kiểm soát và can
thiệp vào các đường truyền thông tin gây nhiễu nhằm cải thiện khả năng tổng quát
hóa ngoài phân phối (OOD).

*Căn cứ chọn mục tiêu: trên Fakeddit, toàn bộ 19 cộng đồng huấn luyện đều thuần
nhãn 100% (toàn thật hoặc toàn giả) — GNN thông thường chỉ cần học "bài đăng ở
đâu" là đạt 91% nội phân phối, nhưng sụp còn 52.1% (AUC 0.511 ≈ ngẫu nhiên) khi
tương quan cộng đồng–nhãn đảo ngược. Đây là chế độ thất bại im lặng nguy hiểm
nhất khi triển khai thật.*

**Bốn mục tiêu cụ thể:**

1. **Đề xuất CausalHeteroGNN** — kiến trúc HGNN hai nhánh dựa trên Mô hình Nhân
   quả Cấu trúc (SCM), dùng **can thiệp cấu trúc** (cắt toàn bộ cạnh chạm nút
   Subreddit ở nhánh nhân quả, chặn đường cửa sau X ← C₁ → Y ngay trong đồ thị
   lan truyền thông điệp) kết hợp Gradient Reversal Layer và ràng buộc trực giao.
2. **Xây dựng hệ đánh giá ba lớp** đo đúng tính bền vững: (i) lớp *tự nhiên* —
   Leave-One-Community-Out 4 folds + temporal split; (ii) lớp *can thiệp* —
   counterfactual do(·) trên từng bài; (iii) lớp *stress-test có kiểm soát* —
   Confounding-Shift Benchmark (đảo tương quan ρ=0.9→0.1, kiểu ColoredMNIST);
   cùng bộ baseline huấn luyện độc lập (ERM, MLP content-only, IRM, EERM).
3. **Bảo đảm tính trung thực của đánh giá OOD**: nhận diện và loại trừ rò rỉ
   nhãn (FastRP, CROSS_POST, thống kê node); phân rã tường minh nguồn tín hiệu
   của mô hình bằng ablation; báo cáo worst-group và đa seed.

*(AutoCut — quy trình tự khám phá quan hệ cần cắt — thuộc Phương pháp nghiên cứu
(mục 4 phần III) và Nội dung nghiên cứu, không tách thành mục tiêu riêng.)*

---

## 2. NỘI DUNG NGHIÊN CỨU

1. **Thu thập và biểu diễn dữ liệu đồ thị (Neo4j):**
   - Lọc 5,898 bài Fakeddit (2008–2020) có ảnh thật; chia train 5,000 (cân bằng
     50/50) / validation 400 / test 498 (200 seen + 298 OOD).
   - Mô hình hóa thành đồ thị thuộc tính: 17,079 nút (5,898 Post; 4,604 User;
     658 Domain; 21 Subreddit; 5,898 Image), 28,274 quan hệ thuộc 5 loại
     (POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE, MEMBER_OF).
   - Import Neo4j (constraint, batch UNWIND/MERGE); chạy GDS: PageRank, Louvain,
     Betweenness, FastRP, Node Similarity — phục vụ phân tích và BI dashboard.
2. **Trích xuất đặc trưng đa phương thức (chống rò rỉ):**
   - Tiêu đề → SentenceTransformer all-mpnet-base-v2 (768-d, chuẩn hóa L2).
   - Ảnh → CLIP ViT-B/32 (512-d).
   - Đặc trưng **nhất quán tiêu đề–ảnh**: cosine(CLIP-text, CLIP-image) — tín
     hiệu lệch ngữ nghĩa của tin giả (cos thật 0.295 vs giả 0.245).
   - User/Subreddit/Domain: thống kê hành vi **chỉ tính từ tập huấn luyện**;
     nút chưa thấy nhận giá trị trung tính 0.5.
3. **Thiết kế và huấn luyện mô hình hai nhánh** (encoder HeteroGraphSAGE chung
   96-d; nhánh baseline trên đồ thị đầy đủ / nhánh nhân quả trên đồ thị đã cắt;
   GRL α=2.0; trực giao; multi-task 2-way + 6-way; early stopping theo val loss).
4. **Xây dựng các giao thức đánh giá và bộ baseline** (ba lớp như Mục tiêu 2;
   4 baseline cùng backbone đại diện 4 chiến lược: bỏ mặc / phạt mềm env-cho-trước /
   phạt mềm env-tự-sinh / né đồ thị).
5. **Quy trình AutoCut**: duyệt 7 ứng viên cắt quan hệ; tiêu chí chọn = độ chính
   xác probe tuyến tính đoán môi trường từ biểu diễn nhân quả, tính trên
   validation thuộc phân phối huấn luyện.
6. **Phân tích chuyên sâu và sản phẩm**: rò rỉ FastRP; counterfactual do(·) +
   Label-Flip Rate; ablation lịch sử nguồn tin; worst-group accuracy; BI
   dashboard (Streamlit + Neo4j); pipeline tái lập đầy đủ trên CPU.

---

## 3. PHƯƠNG PHÁP NGHIÊN CỨU

1. **Suy luận nhân quả:** khai báo SCM với biến nội dung X, ảnh I, nguồn tin D,
   nhãn Y; hai confounder C₁=Subreddit, C₂=User. Chặn đường cửa sau X←C₁→Y bằng
   **phẫu thuật đồ thị** (graph surgery) — phân biệt tường minh với hiệu chỉnh
   cửa sau đầy đủ (không phân tầng theo phân phối confounder).
   *Vì sao cắt cứng thay vì phạt mềm: khi tương quan confounder–nhãn quá mạnh
   (tự nhiên = 1.0), ràng buộc mềm bị "mua chuộc" bởi lợi ích phân loại — kiểm
   chứng bằng IRM/EERM đều không vượt trần content.*
2. **Học sâu trên đồ thị không đồng nhất:** HeteroGraphSAGE 2 lớp, mỗi loại quan hệ một
   bộ trọng số SAGEConv, các modality hợp nhất qua lan truyền thông điệp (không
   nối vector thô).
3. **Học biểu diễn phân rã:** đối kháng qua GRL (DANN) + ràng buộc trực giao
   giữa biểu diễn nhân quả và biểu diễn giả.
4. **Khám phá confounder bằng tìm kiếm cấu trúc:** thay vì gradient (đã kiểm
   chứng thất bại — encoder hấp thụ áp lực đối kháng), duyệt rời rạc các ứng
   viên cắt và chọn bằng env-probe trên validation — tiêu chí scale-free, không
   oracle (chuẩn training-domain validation).
5. **Thực nghiệm đối chứng nghiêm ngặt:**
   - Cùng backbone/siêu tham số/cách chia cho mọi phương pháp; biến số duy nhất
     là cơ chế xử lý confounder.
   - **3 random seeds** cho toàn ma trận, **5 seeds** cho dòng chủ lực; báo cáo
     mean±std; *quy tắc kết luận: chỉ khẳng định khi thứ tự nhất quán qua mọi
     seed; |Δ| < tổng std → tuyên bố "hòa trong nhiễu"*.
   - Chọn mô hình bằng validation cùng phân phối huấn luyện (không oracle).
6. **Chế độ suy luận đúng mục đích:** inductive content-only cho lớp tự nhiên
   (che mọi cạnh chạm bài test — loại lối tắt thành viên-cộng đồng vốn thổi
   accuracy lên ~99%); transductive cho stress-test (mô hình phải nhìn thấy
   confounder bị đảo thì phép thử mới có nghĩa).
7. **Công nghệ:** Python/PyTorch Geometric, Neo4j 5.12 + GDS (Docker),
   Streamlit; toàn bộ chạy và tái lập trên CPU phổ thông.

---

## 4. KẾT QUẢ NGHIÊN CỨU

### 4.1. Kết quả chính trên ba lớp đánh giá

**Lớp stress-test — Confounding-Shift (3 seeds; mô hình đầy đủ 5 seeds):**

| Mô hình | Seen Acc% | OOD Acc% | AUC | F1-Drop | Worst-Group% |
|---|---|---|---|---|---|
| Baseline GNN (ERM) | 91.0±0.4 | 52.1±7.8 | 0.511 | 45.1% | 23.5 |
| IRM | 91.2±0.2 | 54.1±4.7 | 0.524 | 43.2% | 24.4 |
| EERM | 93.2±0.2 | 59.6±1.1 | 0.694 | 38.6% | 28.6 |
| MLP (content-only) | 81.7±0.6 | 59.5±1.7 | 0.685 | 30.0% | 15.6 |
| **CausalHeteroGNN** | 83.8±0.5 | **74.2±3.6** | **0.851** | **12.7%** | **37.8** |
| **+ đặc trưng nhất quán CLIP (5 seeds)** | 84.2±0.9 | **79.9±4.2** | **0.922** | **5.4%** | **56.0** |

→ Cắt cấu trúc loại confounder triệt để hơn phạt mềm; thứ tự nhất quán cả 3 seeds.

**Lớp tự nhiên — LOCO 3 folds cộng đồng thật (seed 42):**

| Mô hình | OOD Acc% | Macro-F1 |
|---|---|---|
| ERM (GNN thuần) | 56.5±1.3 | **0.403 — sụp ở mọi fold** |
| **CausalHeteroGNN** | **69.4±5.4** | 0.677 |
| MLP | 71.6±7.9 | 0.706 |

→ **Lối tắt cộng đồng gây hại trên dữ liệu thật**, không chỉ trong benchmark
tổng hợp; mô hình đề xuất giữ được đồ thị mà không bị nó phản chủ, tiệm cận trần
content. Trên held-out gốc (5 seeds): 59.6±1.9 ≈ MLP 60.3±0.4 (hòa), worst-group
42.9 cao nhất, nhóm satire cải thiện 30%→43%.

**Temporal split** (OOD = 10% bài mới nhất): AUC ba mô hình ≈ nhau (0.66–0.69)
— dịch chuyển thời gian của Fakeddit chủ yếu là dịch chuyển tiên nghiệm nhãn;
chưa mô hình nào xử lý được → giới hạn chung, hướng nghiên cứu tiếp.

**Lớp can thiệp — Label-Flip Rate:** do(C₁=neutral): baseline 30.1% vs causal
≈0.0% (kiểm chứng cơ chế — bất biến theo thiết kế); do(I=∅): 4.0→8.0%;
do(D=credible): 6.2→15.5%.

### 4.2. Kết quả khám phá confounder (AutoCut)

- Trên benchmark confounded: **3/3 seeds tự chọn đúng cặp {posted_in, member_of}**
  (= cô lập Subreddit) với probe 0.755–0.777, tách biệt rõ mọi ứng viên sai
  (0.93–1.00); OOD của phép cắt được chọn 76.8–86.6%.
- Trên dữ liệu tự nhiên không có confounder trội: lựa chọn bất ổn định với
  khoảng cách probe nhỏ → quy trình "từ chối khám phá" thay vì cắt bừa.
- Hai kết quả âm tính công bố minh bạch: cổng gradient thất bại; tiêu chí
  IRM-penalty chọn sai.

### 4.3. Kết quả về tính trung thực của đánh giá

- **Rò rỉ FastRP**: giữ FastRP → OOD "ảo" 93.6–94.3%; loại bỏ → 61.0–61.1%
  (chênh +33 điểm hoàn toàn do rò rỉ cấu trúc).
- **Phân rã nguồn tín hiệu**: trung hòa lịch sử uy tín domain → OOD giảm
  74.2%→58.8% — ~15.4 điểm đến từ lịch sử nguồn, phần còn lại trùng trần
  content-only; công bố chủ động.
- GroupDRO không cải thiện (55.7 vs 56.7) — thử và loại theo đúng quy trình.

### 4.4. Sản phẩm

1. Mã nguồn pipeline tái lập đầy đủ (10 script chính + script split/figure),
   chạy trên CPU; hướng dẫn `RUNNING_GUIDE.md`.
2. HIN Fakeddit trên Neo4j (schema 5 nút/5 quan hệ + kết quả GDS) và BI
   dashboard Streamlit (trực quan kết quả, counterfactual, giải thích dự đoán).
3. Bộ benchmark + 6 phương án chia dữ liệu (gốc, confounding-shift, temporal,
   3 LOCO folds) và toàn bộ kết quả dạng JSON trace được (`results/`).
4. Bài báo EN + VN (đã revision) và bộ tài liệu: `STATS_MASTER.md`,
   `IMPROVEMENTS.md`, `EVAL_REALISM.md`, `RESPONSE_TO_COMMITTEE.md`.

### 4.5. Giới hạn đã khai báo

Nhận diện satire bằng nội dung vẫn <50% dù đã cải thiện; dịch chuyển tiên nghiệm
nhãn theo thời gian chưa xử lý được (giới hạn chung mọi mô hình); phạm vi một bộ
dữ liệu, OOD set 298–822 bài/protocol; ~15.4 điểm của kết quả chính dựa vào đặc
trưng lịch sử nguồn (suy giảm nếu phân phối domain đổi).
