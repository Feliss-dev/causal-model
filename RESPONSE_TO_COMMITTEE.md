# PHẢN HỒI GÓP Ý HỘI ĐỒNG
### Đề tài: Causal Graph Disentanglement with Heterogeneous GraphSAGE for Multimodal Misinformation Detection

> Tài liệu này trả lời lại 4 nhóm câu hỏi của hội đồng theo đúng tinh thần
> "nêu vấn đề nào, trả lời trực tiếp vấn đề đó". Mọi số liệu trích dẫn đều tái
> lập được từ `results/final_tables.md` (mean±std trên 3 seeds) và có lệnh chạy
> tương ứng trong `RUNNING_GUIDE.md`.

## Cách đọc tài liệu này

- Mỗi mục bắt đầu bằng **câu hỏi gốc của hội đồng**.
- Ngay sau đó là phần **trả lời ngắn gọn**, có thể dùng gần như nguyên văn khi
  phản hồi.
- Các tiểu mục phía dưới cung cấp **giải thích kỹ thuật, căn cứ số liệu và các
  chỉnh sửa đã bổ sung vào bản sửa bài**.

---

## CÂU HỎI 1 — OOD thể hiện như thế nào trong bài? Tại sao chuẩn bị dữ liệu theo OOD? Việc chuẩn bị dữ liệu đã đáp ứng OOD chưa?

> **Câu hỏi gốc của hội đồng:** "OOD thể hiện như thế nào trong bài? Tại sao phải
> chuẩn bị dữ liệu theo OOD? Việc chuẩn bị dữ liệu như vậy đã thực sự đáp ứng
> đánh giá OOD chưa?"

### Trả lời ngắn gọn

OOD là trọng tâm khoa học của bài vì bài toán thực tế không phải là dự đoán tốt
trên đúng phân phối đã thấy, mà là **vẫn dự đoán đúng khi sang cộng đồng hoặc
điều kiện phân phối mới**. Trong Fakeddit, subreddit gần như quyết định nhãn, nên
subreddit trở thành **biến gây nhiễu** rất mạnh. Nếu chỉ đánh giá IID, mô hình có
thể đạt điểm cao nhờ học "đường tắt cộng đồng" thay vì học nội dung. Vì vậy, việc
chuẩn bị dữ liệu theo OOD là cần thiết để kiểm tra đúng câu hỏi nghiên cứu: mô
hình có học tín hiệu ổn định hay không. Chúng tôi đã dùng **hai giao thức OOD bổ
trợ nhau** và bổ sung nhiều lớp chống rò rỉ để bảo đảm kết quả phản ánh OOD thật,
không phải IID trá hình.

### 1.1. OOD trong bài là gì và thể hiện ở đâu

**OOD (Out-of-Distribution)** = tình huống dữ liệu kiểm thử đến từ **phân phối khác**
với dữ liệu huấn luyện. Với bài toán phát hiện tin giả, đây không phải lựa chọn
tùy ý mà là **điều kiện vận hành thực tế**: hệ thống luôn phải xử lý bài đăng từ
cộng đồng mới, chủ đề mới, nguồn tin mới chưa từng xuất hiện khi huấn luyện. Một
mô hình chỉ được kiểm thử trên cùng phân phối (IID) sẽ cho con số đẹp nhưng
**không nói lên gì về khả năng triển khai**.

Trong bài, OOD được hiện thực hóa bằng **hai giao thức bổ trợ nhau** (Mục 5.1):

| Giao thức | Loại dịch chuyển | Cách tạo | Đo điều gì |
|---|---|---|---|
| **(i) Held-Out-Domain** | Dịch chuyển cộng đồng/chủ đề **tự nhiên** | 2 subreddit `r/neutralnews` (toàn Real) và `r/theonion` (toàn Fake) bị **loại hoàn toàn** khỏi train/val, chỉ dùng làm test | Khả năng tổng quát hóa sang cộng đồng chưa từng thấy |
| **(ii) Confounding-Shift** | Dịch chuyển tương quan **có kiểm soát** (kiểu ColoredMNIST) | Thay subreddit thật bằng biến gây nhiễu nhị phân: train tương quan với nhãn ρ=0.9, OOD test **đảo ngược** ρ=0.1 | Mô hình dựa vào nội dung hay dựa vào lối tắt cộng đồng |

**Bằng chứng OOD là thật, không phải IID trá hình:**
- Giao thức (i): độ chính xác seen-test 76–82% nhưng OOD chỉ 57–61% ở *mọi* mô hình
  → tồn tại khoảng cách phân phối thực sự (~20 điểm).
- Giao thức (ii): ERM đạt 91.0% in-distribution nhưng sụp còn **52.1% với AUC 0.511**
  khi tương quan đảo — mất hoàn toàn khả năng phân biệt. Nhánh baseline diagnostic
  thậm chí cho **AUC 0.314 < 0.5** (biên quyết định bị đảo ngược hệ thống). Đây là
  "chữ ký" định lượng của việc mô hình học lối tắt — điều chỉ bộc lộ được dưới OOD.

### 1.2. Tại sao phải chuẩn bị tập dữ liệu theo OOD

Vì **câu hỏi nghiên cứu** của đề tài không phải "mô hình đạt bao nhiêu % trên
Fakeddit" mà là "**mô hình có học cơ chế ổn định thay vì tương quan giả không**".
Câu hỏi này về nguyên tắc **không thể trả lời bằng kiểm thử IID**: trên phân phối
huấn luyện, mô hình dựa lối tắt và mô hình hiểu nội dung cho kết quả gần như nhau
(91–93%). Chỉ khi phân phối dịch chuyển — đặc biệt khi tương quan confounder–nhãn
bị đảo — hai loại mô hình mới tách nhau ra (52.1% vs 74.2%). Giao thức (ii) được
thiết kế đúng theo chuẩn của cộng đồng nghiên cứu OOD (ColoredMNIST của Arjovsky
et al. [8]; DomainBed của Gulrajani & Lopez-Paz [10]).

### 1.3. Việc chuẩn bị dữ liệu đã đáp ứng OOD chưa — rà soát từng lớp chống rò rỉ

Chúng tôi đã rà soát lại toàn bộ `01_prepare_data.py` theo góp ý. Các biện pháp
bảo đảm tính OOD và chống rò rỉ (leakage) gồm **6 lớp**:

1. **Tách tuyệt đối subreddit OOD**: `neutralnews`, `theonion` bị loại khỏi train
   và validation ngay từ bước lọc (`01_prepare_data.py`, hàm
   `process_and_sample_dataset`) — không một bài nào của 2 cộng đồng này xuất hiện
   trong huấn luyện hay chọn siêu tham số.
2. **Đặc trưng thống kê chỉ tính từ train**: mọi đặc trưng hành vi (fake_rate của
   User, fake_ratio của Subreddit/Domain, post_count…) được tổng hợp **chỉ trên
   tập train**; nút chưa từng thấy trong train nhận giá trị trung tính 0.5
   (`compute_real_features`).
3. **Loại FastRP khỏi đầu vào mô hình** (Mục 5.4): FastRP do Neo4j GDS tính trên
   *toàn* đồ thị nên mã hóa cụm cấu trúc ≈ nhãn của bài OOD. Giữ FastRP cho OOD
   "ảo" 94%; loại ra còn ~61% — chúng tôi chủ động phát hiện, báo cáo và loại bỏ
   nguồn rò rỉ này.
4. **Đánh giá quy nạp (inductive) cho giao thức (i)**: khi suy luận, *mọi cạnh*
   chạm bài test bị che — bài test được phân loại chỉ từ đặc trưng của chính nó,
   loại bỏ lối tắt "thành viên của subreddit" vốn thổi phồng độ chính xác
   transductive (`04_evaluate.py`, hàm `mask_post_edges`).
5. **Loại bỏ các cấu phần gây leak khác**: cạnh CROSS_POST (nối bài cùng nhãn) và
   nút Comment (không có dữ liệu thật) đã bị xóa khỏi pipeline.
6. **Chọn mô hình không dùng oracle**: validation của giao thức (ii) tuân theo
   tương quan huấn luyện (ρ=0.9) — không giả định truy cập phân phối đã dịch chuyển.

**Điểm hội đồng góp ý đúng và đã bổ sung trong bản chỉnh sửa**: đặc trưng lịch sử
nguồn tin (Domain fake-ratio) không bị cắt bởi can thiệp Subreddit. Chúng tôi đã
bổ sung **Mục 5.6 (ablation)**: trung hòa đặc trưng này làm OOD giảm 74.2% → 58.8%,
tức ~15.4 điểm của kết quả đến từ lịch sử nguồn — được khai báo tường minh thay vì
trình bày 74.2% như tổng quát hóa thuần túy từ nội dung.

---

## CÂU HỎI 2 — Làm rõ luồng xử lý dữ liệu ở Hình 2: từng loại dữ liệu xử lý thế nào, đầu ra là gì, kết hợp ra sao? Các khối B, C₁, D là gì? Tại sao là C₁ mà không phải C?

> **Câu hỏi gốc của hội đồng:** "Hình 2 đang đi khá nhanh. Cần làm rõ từng loại
> dữ liệu đi qua khối nào, đầu ra là gì, kết hợp với nhau ra sao. Đồng thời giải
> thích rõ các khối B, C₁, D và vì sao dùng ký hiệu C₁ chứ không phải C."

### Trả lời ngắn gọn

Hình 2 mô tả một pipeline gồm ba lớp ý chính: **trích đặc trưng theo từng loại
dữ liệu**, **đưa các loại dữ liệu vào cùng một đồ thị dị thể để trộn bằng
message-passing**, và **tách thành nhánh baseline với nhánh nhân quả**. Ký hiệu
**C₁** được dùng vì trong mô hình nhân quả của bài có **nhiều confounder**, trong
đó Subreddit là confounder thứ nhất cần bị cắt, còn User là confounder thứ hai.
Phần chỉnh sửa của chúng tôi đã bổ sung rõ đầu vào, số chiều đầu ra, vai trò của
từng khối và cách các modality được kết hợp.

### 2.1. Trả lời ngay câu "tại sao C₁ mà không phải C"

**C₁ là ký hiệu có chỉ số trong Mô hình Nhân quả Cấu trúc (SCM)** của bài, vì đồ
thị nhân quả có **nhiều hơn một biến gây nhiễu (confounder)**:

- **C₁ = Subreddit** (cộng đồng đăng tải) — confounder chính, bị xử lý bằng can
  thiệp cấu trúc.
- **C₂ = User** (người đăng) — confounder thứ hai được khai báo trong DAG nhưng
  *không* bị cắt (lý do và hệ quả được định lượng ở Mục 5.6).

Các biến còn lại trong SCM: **X** = nội dung bài (text), **I** = ảnh, **D** = nguồn
tin (Domain), **Y** = nhãn thật/giả. Đường nhân quả: X→Y, I→Y, D→Y. Đường cửa sau
(backdoor) cần chặn: **X ← C₁ → Y**. Viết "C" trống sẽ không phân biệt được hai
confounder — chỉ số ₁, ₂ là quy ước chuẩn của tài liệu nhân quả (Pearl [2]).

### 2.2. Từng loại dữ liệu được xử lý thế nào — đầu vào, phép biến đổi, đầu ra

| Loại nút | Dữ liệu thô | Phép xử lý | Vector đầu ra |
|---|---|---|---|
| **Post** | tiêu đề bài + 3 chỉ số (score, upvote_ratio, num_comments) | tiêu đề → SentenceTransformer `all-mpnet-base-v2` (chuẩn hóa L2); 3 chỉ số → min-max | **771-d** = 768 (text) + 3 (scalar) |
| **Image** | ảnh thumbnail tải từ URL | CLIP ViT-B/32 → vision encoder + projection, chuẩn hóa L2 | **512-d** |
| **User** | lịch sử đăng bài *trong tập train* | tổng hợp: post_count, avg_score, avg_upvote_ratio, fake_rate | **4-d** |
| **Subreddit** | lịch sử cộng đồng *trong train* | post_count, fake_ratio_real, avg_score | **3-d** |
| **Domain** | lịch sử nguồn tin *trong train* | fake_ratio_real, avg_upvote_ratio, post_count | **3-d** |

(FastRP 64-d **không** nằm trong đầu vào mặc định — chỉ dùng cho phân tích rò rỉ
Mục 5.4 và BI dashboard.)

### 2.3. Các đầu ra khác nhau kết hợp với nhau như thế nào

Đây là điểm quan trọng cần làm rõ: các modality **không được nối (concatenate)
trực tiếp** thành một vector. Chúng kết hợp qua **3 tầng**:

**Tầng 1 — Chiếu về không gian chung (khối Projection):** mỗi loại nút có chiều
khác nhau (771/512/4/3/3) nên đi qua một phép biến đổi tuyến tính *riêng* để về
cùng không gian ẩn **d = 96**, theo sau là ReLU + dropout 0.4. Sau tầng này, mọi
nút — dù là bài viết, ảnh hay người dùng — đều là vector 96 chiều so sánh được.

**Tầng 2 — Lan truyền thông điệp dị thể (khối HeteroGraphSAGE ×2):** thông tin
các modality "chảy" vào nhau **dọc theo cạnh của đồ thị**. Mỗi loại quan hệ
(posted_by, posted_in, links_to, has_image, member_of + các cạnh ngược) có một
bộ trọng số SAGEConv riêng; biểu diễn mới của nút Post = trạng thái hiện tại của
nó + tổng các thông điệp từ hàng xóm theo từng loại quan hệ. Sau 2 lớp, vector
96-d của mỗi Post đã **hòa trộn**: ngữ nghĩa tiêu đề (chính nó) + đặc trưng ảnh
(qua cạnh has_image) + uy tín nguồn (qua links_to) + hành vi người đăng (qua
posted_by) + ngữ cảnh cộng đồng (qua posted_in — chỉ ở nhánh baseline).

**Tầng 3 — Hai nhánh và các đầu phân loại:** cùng một bộ encoder (chia sẻ trọng
số) chạy trên **hai phiên bản đồ thị** rồi đi qua các head riêng (chi tiết 2.4).

### 2.4. Giải nghĩa từng khối của Hình 2 (kèm sơ đồ chữ)

```
 [A] INPUT & FEATURE EXTRACTION
     Post 771-d | Image 512-d | User 4-d | Subreddit 3-d | Domain 3-d
              │
 [B] SHARED ENCODER (dùng chung trọng số cho cả hai nhánh)
     Linear chiếu mỗi loại nút → 96-d → ReLU, dropout 0.4
     HeteroGraphSAGE lớp 1 → lớp 2 (mỗi loại cạnh một SAGEConv, gộp sum)
              │ encoder được chạy HAI LẦN trên hai đồ thị:
      ┌───────┴────────────────────────┐
 [C] NHÁNH BASELINE                [C₁-cut] NHÁNH NHÂN QUẢ
     đồ thị G ĐẦY ĐỦ                   đồ thị G_causal = G \ {mọi cạnh chạm
     (nhìn thấy Subreddit = C₁)        nút Subreddit}  ← CAN THIỆP CẤU TRÚC
              │                        (chặn đường cửa sau X ← C₁ → Y)
        h_post (96-d)                     h_post_causal (96-d)
              │                                 │
       spurious_head → h_s              causal_head → h_c
              │                                 │
 [D] CÁC RÀNG BUỘC PHÂN RÃ (disentanglement)
     • L_ortho = |cos(h_c, h_s)| → ép hai biểu diễn không trùng thông tin
     • GRL(α=2.0) trên h_c → bộ phân loại subreddit-ID:
       gradient bị ĐẢO DẤU khi lan ngược → h_c bị ép KHÔNG đoán được
       bài thuộc cộng đồng nào (xóa rò rỉ gián tiếp của C₁)
     • L_spurious: h_s được ép PHẢI đoán được subreddit (hấp thụ tín hiệu giả)
              │
 [E] PHÂN LOẠI & TỔN THẤT
     mỗi nhánh có head 2-way (Real/Fake) + 6-way (loại tin giả chi tiết)
     L = L_base,2w + 0.5·L_base,6w + L_causal,2w + 0.5·L_causal,6w
         + 0.5·L_spurious + 0.5·L_adv + 0.2·L_ortho
     → Dự đoán cuối cùng dùng cho OOD = đầu ra NHÁNH NHÂN QUẢ
```

**Vì sao cần cả hai nhánh?** Nhánh baseline + spurious_head đóng vai trò "bể chứa"
hấp thụ tín hiệu cộng đồng; ràng buộc trực giao và GRL đẩy tín hiệu đó **ra khỏi**
h_c. Nếu chỉ có một nhánh và một ràng buộc phạt mềm, thí nghiệm của chúng tôi cho
thấy không đủ (GRL đơn lẻ thất bại trên benchmark confounding-shift) — phải có
**cắt cấu trúc** thì nhánh nhân quả mới thực sự sạch tín hiệu C₁ (bằng chứng:
LFR do(C₁) ≈ 0.0%, Bảng 2).

**Tiếp thu góp ý:** chúng tôi đã bổ sung mô tả từng khối với số chiều cụ thể vào
Mục 4 của bài và chú thích Hình 2; sơ đồ giải nghĩa trên đây có thể đưa vào phụ
lục hoặc thay Hình 2 bằng phiên bản chi tiết hơn nếu hội đồng yêu cầu.

---

## CÂU HỎI 3 — Các mô hình so sánh có liên quan không? Tại sao chọn IRM, EERM? Chúng có được tạo ra để giải quyết OOD không? Số liệu thấp đi thì so sánh để làm gì, có công bằng không?

> **Câu hỏi gốc của hội đồng:** "IRM, EERM có liên quan trực tiếp đến bài toán
> này không, hay chỉ đưa vào để so sánh cho đủ? Chúng có thực sự là các phương
> pháp OOD không? Nếu kết quả của chúng thấp đi thì phép so sánh đó còn ý nghĩa
> hay công bằng không?"

### Trả lời ngắn gọn

IRM và EERM là các baseline **rất liên quan**, vì chúng đều được thiết kế để xử
lý **tổng quát hóa OOD**. IRM là phương pháp kinh điển của học bất biến, còn EERM
là một baseline OOD dành riêng cho đồ thị/GNN. Chúng tôi chọn chúng không phải để
"làm nền" cho mô hình đề xuất, mà để so sánh **hai triết lý xử lý confounder**:
phạt mềm bằng ràng buộc bất biến so với can thiệp cấu trúc bằng cắt cạnh. Việc số
OOD của các baseline thấp hơn không làm so sánh mất ý nghĩa; ngược lại, đó chính
là bằng chứng cho thấy các phương pháp phạt mềm vẫn còn phụ thuộc vào đường tắt,
trong khi can thiệp cấu trúc bền hơn dưới dịch chuyển phân phối mạnh.

### 3.1. IRM và EERM có phải phương pháp OOD không — Có, đúng chuẩn lĩnh vực

- **IRM — Invariant Risk Minimization** (Arjovsky, Bottou, Gulrajani, Lopez-Paz,
  2019 [8]): là **phương pháp kinh điển nhất** của hướng học bất biến cho OOD;
  chính nhóm tác giả này tạo ra ColoredMNIST — nguyên mẫu mà Confounding-Shift
  Benchmark của chúng tôi mô phỏng. Ý tưởng: phạt mô hình nếu bộ phân loại tối ưu
  *khác nhau giữa các môi trường* → ép học đặc trưng ổn định qua môi trường.
  Trong bài, "môi trường" = phân hoạch theo subreddit — đúng chiều confounder.
- **EERM — Explore-to-Extrapolate Risk Minimization** (Wu, Zhang, Yan, Wipf,
  ICLR 2022 [13]): là phương pháp OOD **được thiết kế riêng cho đồ thị/GNN** —
  sinh nhiều môi trường ảo bằng chỉnh sửa cạnh rồi phạt phương sai rủi ro giữa
  các môi trường. Đây là baseline graph-OOD tiêu chuẩn trong các bài cùng hướng.

Như vậy cả hai **được tạo ra đúng để giải quyết OOD**, và đại diện cho **họ giải
pháp đối lập** với đề xuất của chúng tôi: *ràng buộc phạt mềm (soft invariance
penalty)* so với *can thiệp cấu trúc cứng (hard structural cut)*. So sánh này
không phải chọn ngẫu nhiên mô hình mạnh-yếu, mà là **so sánh hai triết lý xử lý
cùng một vấn đề** — đó chính là câu hỏi khoa học của bài.

### 3.2. "Số liệu thấp đi vậy so sánh để làm gì?" — Số thấp chính là phát hiện

Cần tách hai cột số liệu:

| Mô hình | In-distribution (Seen) | OOD khi tương quan đảo |
|---|---|---|
| Baseline GNN (ERM) | 91.0±0.4 | **52.1±7.8** (AUC 0.511 ≈ ngẫu nhiên) |
| IRM | 91.2±0.2 | 54.1±4.7 |
| EERM | 93.2±0.2 | 59.6±1.1 |
| MLP (content-only, không đồ thị) | 81.7±0.6 | 59.5±1.7 |
| **CausalHeteroGNN (đề xuất)** | 83.8±0.5 | **74.2±3.6** (AUC 0.851) |

- Trên in-distribution, *mọi* mô hình đều ~91–93% — nếu chỉ nhìn cột này thì
  "không có gì để nghiên cứu". **Mục đích của benchmark là làm các con số thấp
  xuống một cách có kiểm soát** để lộ ra mô hình nào dựa lối tắt: ERM/IRM/EERM
  rơi 32–39 điểm, mô hình đề xuất chỉ rơi ~10 điểm. Cột OOD thấp không phải khuyết
  điểm của thí nghiệm — nó **là kết quả đo lường** mức độ phụ thuộc lối tắt.
- Thứ tự **ERM 52.1 < IRM 54.1 < EERM 59.6 < Causal 74.2 nhất quán trên cả 3
  seeds**, cộng worst-group accuracy (Causal 37.8% so với ≤28.6% của các phương
  pháp còn lại) → kết luận "cắt cứng triệt để hơn phạt mềm" có bằng chứng lặp lại.

### 3.3. So sánh có công bằng không — các biện pháp bảo đảm

1. **Cùng backbone tuyệt đối**: IRM, EERM, ERM dùng *chính* bộ mã hóa
   HeteroGraphSAGE của mô hình đề xuất (cùng projection, cùng 2 lớp SAGE, cùng
   hidden 96, dropout 0.4, learning rate, weight decay, early stopping theo val
   loss, gradient clipping). Khác biệt duy nhất là cơ chế bất biến — đúng biến số
   cần so sánh (`06_baselines_irm_eerm.py`, `07_baselines_erm_mlp.py`).
2. **Huấn luyện độc lập**: tiếp thu góp ý, trong bản chỉnh sửa chúng tôi đã thay
   "Baseline GNN" cũ (vốn là nhánh baseline đồng-huấn-luyện bên trong mô hình đề
   xuất, con số 36.4%) bằng **ERM huấn luyện độc lập hoàn toàn (52.1%)**. Con số
   36.4% chỉ còn được nêu như một chẩn đoán phụ trong Mục 5.3.
3. **Cùng dữ liệu, cùng cách chia, cùng 3 seeds, báo cáo mean±std.**
4. **Có mốc neo dưới (MLP content-only)**: cho biết "trần" đạt được nếu hoàn toàn
   không dùng đồ thị (59.5%) — giúp diễn giải IRM/EERM (không vượt trần này) và
   phần vượt trội của mô hình đề xuất.
5. **Tự khai giới hạn của chính mình**: ablation Mục 5.6 chỉ rõ ~15.4 điểm của
   74.2% đến từ đặc trưng lịch sử domain; phần "thuần nội dung" ≈ 59%, ngang trần
   MLP. Chúng tôi chủ động báo cáo điều này để phép so sánh minh bạch cả hai chiều.

---

## CÂU HỎI 4 — Bản chất mô hình đề xuất là gì? Các bước thế nào? Dữ liệu xử lý ra sao khi đưa vào mô hình? Việc phân chia tập huấn luyện/kiểm thử đã phù hợp chưa?

> **Câu hỏi gốc của hội đồng:** "Bản chất thật sự của mô hình đề xuất là gì? Các
> bước xử lý dữ liệu và huấn luyện diễn ra như thế nào? Khi đưa dữ liệu vào mô
> hình thì từng loại dữ liệu được xử lý ra sao? Cách chia train/validation/test
> hiện tại đã hợp lý chưa?"

### Trả lời ngắn gọn

Bản chất mô hình đề xuất không phải là một lớp tích chập đồ thị mới, mà là
**một HeteroGraphSAGE chuẩn được đặt trong khung huấn luyện nhân quả**. Cụ thể,
mô hình học song song trên một đồ thị đầy đủ và một đồ thị đã cắt confounder
Subreddit, rồi dùng GRL và ràng buộc trực giao để tách tín hiệu ổn định khỏi tín
hiệu giả. Dữ liệu đầu vào được xử lý riêng theo từng modality, chiếu về cùng
không gian ẩn, rồi kết hợp qua message-passing trên đồ thị. Cách chia dữ liệu là
phù hợp cho mục tiêu nghiên cứu OOD, đồng thời chúng tôi cũng đã nêu rõ các giới
hạn còn lại như kích thước OOD-test nhỏ và phương sai theo seed.

### 4.1. Bản chất mô hình đề xuất (nói thẳng, không khoa trương)

CausalHeteroGNN **không phải một toán tử tích chập đồ thị mới**. Bản chất là:

> **Một bộ mã hóa HeteroGraphSAGE chuẩn, được huấn luyện đồng thời trên hai phiên
> bản đồ thị — bản đầy đủ và bản đã phẫu thuật cắt confounder — kèm hai ràng buộc
> phân rã (đối kháng GRL + trực giao) để tách "tín hiệu nội dung ổn định" khỏi
> "tín hiệu cộng đồng giả".**

Điểm mới của bài nằm ở **ba thứ**: (1) cơ chế *can thiệp cấu trúc* — xóa mọi cạnh
chạm Subreddit ở nhánh nhân quả, tương đương chặn đường cửa sau X ← C₁ → Y ngay
trong đồ thị tính toán (khác về chất với các phương pháp phạt mềm: IRM/EERM chỉ
*giảm* phụ thuộc, can thiệp cấu trúc *loại hẳn*); (2) **giao thức đánh giá**
Confounding-Shift cho HIN đa phương thức; (3) các **phân tích trung thực** đi kèm
(rò rỉ FastRP, ablation lịch sử nguồn, worst-group, LFR can thiệp do(·)).

### 4.2. Các bước xử lý dữ liệu khi đưa vào mô hình (tuần tự, đúng theo code)

1. **Lọc & chia** (`01_prepare_data.py`): lọc bài có ảnh thật → tách 2 subreddit
   OOD ra khỏi train/val → sample cân bằng 50/50 Real-Fake.
2. **Trích đặc trưng**: tiêu đề → mpnet 768-d; ảnh → CLIP 512-d; User/Subreddit/
   Domain → thống kê train-only (4/3/3 chiều); chuẩn hóa min-max cho scalar.
3. **Dựng đồ thị** (`02_neo4j_import.py` → `build_heterodata` trong
   `03_train_gnn.py`): 17,079 nút (5,898 Post; 4,604 User; 658 Domain; 21
   Subreddit; 5,898 Image), 28,274 quan hệ thuộc 5 loại + cạnh ngược
   (ToUndirected) → đối tượng PyTorch Geometric `HeteroData`.
4. **Mã hóa**: chiếu từng loại nút → 96-d → 2 lớp HeteroSAGE; chạy **hai lần**:
   trên G (nhánh baseline) và trên G_causal đã cắt cạnh Subreddit (nhánh nhân quả).
5. **Huấn luyện**: tối ưu đồng thời 7 thành phần tổn thất (phân loại 2-way/6-way
   hai nhánh, spurious, đối kháng GRL, trực giao); early stopping theo val loss;
   3 seeds.
6. **Suy luận/đánh giá** (`04_evaluate.py`): giao thức (i) — inductive, che mọi
   cạnh của bài test; giao thức (ii) — transductive (chủ ý, để mô hình đối mặt
   confounder bị đảo); hiệu chỉnh nhiệt độ trên val; can thiệp phản thực
   do(image=∅), do(domain=credible), do(subreddit=neutral) → LFR.

### 4.3. Phân chia dữ liệu — cấu hình và căn cứ

| Tập | Kích thước | Nguồn | Vai trò |
|---|---|---|---|
| Train | 5,000 (2,500 Real + 2,500 Fake) | `multimodal_train.tsv`, ngoài 2 sub OOD | học tham số |
| Validation | 400 (cân bằng) | `multimodal_validate.tsv` (file tách sẵn của Fakeddit), ngoài 2 sub OOD | early stopping + chọn siêu tham số, **không bao giờ** dùng để học |
| Seen-test | 200 (cân bằng) | `multimodal_validate.tsv`, ngoài 2 sub OOD | đo in-distribution |
| OOD-test | 298 (~50% fake) | chỉ từ `neutralnews` + `theonion` | đo tổng quát hóa |

**Căn cứ phù hợp:**
- Train/val/test lấy từ **hai file gốc tách sẵn** của Fakeddit (không tự cắt từ
  một file) → tránh trùng lặp bài giữa các tập.
- Cân bằng 50/50 ở mọi tập → accuracy không bị thiên lệch theo lớp; OOD set ghép
  1 cộng đồng toàn-Real + 1 cộng đồng toàn-Fake để mô hình *không thể* đạt trên
  50% chỉ bằng nhận diện "phong cách cộng đồng" — buộc phải hiểu nội dung.
- Validation chỉ dùng cho chọn mô hình; test (seen + OOD) chỉ dùng đúng một lần
  cho báo cáo cuối.

**Giới hạn đã tự nhận trong bài (và hướng khắc phục):** OOD set 298 bài từ 2 cộng
đồng là nhỏ → phương sai giữa seeds đáng kể (ERM ±7.8); vì vậy mọi kết quả báo
cáo dạng mean±std trên 3 seeds và kết luận chỉ dựa trên thứ tự nhất quán qua các
seeds. Quy mô 5,898 bài là lựa chọn chủ đích để toàn pipeline (Neo4j + GNN + BI)
tái lập được trên máy CPU thông thường; mở rộng số cộng đồng OOD (4–6 subreddit)
và tăng seeds là việc làm tiếp theo đã nêu ở phần Kết luận.

---

## TÓM TẮT CÁC CHỈNH SỬA ĐÃ THỰC HIỆN THEO GÓP Ý

| Góp ý | Hành động | Vị trí |
|---|---|---|
| Làm rõ OOD & chuẩn bị dữ liệu | Bổ sung mô tả 2 chế độ suy luận + chọn mô hình không-oracle; rà soát 6 lớp chống leak | Mục 5.1; tài liệu này §1.3 |
| Làm rõ luồng Hình 2, ký hiệu C₁ | Bổ sung số chiều từng loại nút, vai trò từng khối, định nghĩa C₁/C₂ trong SCM; sơ đồ giải nghĩa | Mục 3–4; tài liệu này §2 |
| So sánh công bằng | Thay baseline bằng **ERM huấn luyện độc lập** (52.1%), thêm **MLP content-only** (mốc neo), giữ IRM/EERM cùng backbone; thêm worst-group accuracy | Mục 5.2–5.3, Bảng 1 |
| Bản chất mô hình & tính phù hợp dữ liệu | Viết lại Mục 4 theo ngôn ngữ "can thiệp cấu trúc"; thêm Mục 5.6 ablation định lượng nguồn tín hiệu; nêu giới hạn OOD set nhỏ | Mục 4, 5.6, 6 |

> Mọi con số trong tài liệu này lấy từ `results/final_tables.md`; lệnh tái lập
> từng bảng nằm trong `RUNNING_GUIDE.md` (Phần E).
