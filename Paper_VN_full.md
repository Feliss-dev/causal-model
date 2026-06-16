# Phân rã quan hệ nhân quả trên đồ thị không đồng nhất bằng GraphSAGE cho phát hiện tin giả đa phương thức
### Causal Graph Disentanglement with Heterogeneous GraphSAGE for Multimodal Misinformation Detection

> **Ghi chú phiên bản.** Bản thảo này là bản cải tiến của các bản prerevision
> tiếng Việt và tiếng Anh: `(VN)_Causal Graph Disentanglement with Heterogeneous
> GraphSAGE for Multimodal Misinformation Detection_BACKUP_prerevision.docx` và
> `(EN)_Causal Graph Disentanglement with Heterogeneous GraphSAGE for Multimodal
> Misinformation Detection_BACKUP_prerevision.docx`. Trong bản này,
> CausalHeteroGNN được trình bày trực tiếp như cấu hình hoàn chỉnh với đặc trưng
> nhất quán CLIP; các kết quả không dùng đặc trưng này chỉ còn là ablation, không
> được trình bày như một phiên bản mô hình riêng.

## Tóm tắt

Phát hiện tin giả đa phương thức trên mạng thông tin không đồng nhất có thể hưởng lợi từ ngữ cảnh quan hệ, nhưng chính ngữ cảnh này cũng có thể tạo ra các lối tắt không bền vững. Trên Fakeddit, Subreddit là một ví dụ cực đoan: trong mẫu nghiên cứu, toàn bộ 19 cộng đồng huấn luyện đều thuần nhãn, khiến mô hình đồ thị có thể đạt độ chính xác cao chỉ bằng cách nhận ra nơi bài đăng xuất hiện thay vì học nội dung của bài. Khi tương quan cộng đồng-nhãn bị đảo trong thiết lập Confounding-Shift, ERM HeteroGraphSAGE đạt 90.6% accuracy trên seen test nhưng giảm còn 55.0% OOD accuracy, với AUC 0.551 gần ngẫu nhiên.

Nghiên cứu này đề xuất CausalHeteroGNN, một mô hình HeteroGraphSAGE hai nhánh nhằm giảm phụ thuộc vào lối tắt cộng đồng. Nhánh baseline chạy trên đồ thị đầy đủ để chẩn đoán mức độ khai thác shortcut, trong khi nhánh nhân quả thực hiện can thiệp cấu trúc bằng cách loại bỏ toàn bộ đường truyền thông điệp chạm nút Subreddit. Mô hình kết hợp Gradient Reversal Layer và ràng buộc trực giao để làm yếu thêm tín hiệu cộng đồng còn sót trong biểu diễn nhân quả. Bên cạnh đó, nghiên cứu đề xuất AutoCut, một quy trình chỉ dùng validation trong phân phối huấn luyện để tự tìm quan hệ gây nhiễu cần cắt.

Hệ đánh giá gồm ba lớp: stress-test Confounding-Shift có kiểm soát, can thiệp phản thực, và kiểm chứng tự nhiên bằng LOCO/temporal. Trên Confounding-Shift, CausalHeteroGNN đạt 79.9% OOD accuracy và AUC 0.922, vượt ERM, IRM, EERM và MLP content-only. Trên LOCO tự nhiên, mô hình vượt rõ ERM GNN và tiệm cận mức content-only. Nghiên cứu cũng chỉ ra FastRP có thể gây rò rỉ nhãn hơn 33 điểm OOD và định lượng phần đóng góp của lịch sử uy tín nguồn tin. Kết quả cho thấy confounding cấu trúc là một trở ngại trung tâm của phát hiện tin giả bằng GNN, và can thiệp ở mức lược đồ quan hệ là một hướng hiệu quả để cải thiện tổng quát hóa ngoài phân phối.

**Từ khóa:** heterogeneous graph neural networks, causal intervention, OOD generalization, multimodal misinformation detection, Fakeddit, GraphSAGE.

## 1. Giới thiệu

Tin giả đa phương thức ngày càng khó phát hiện vì tín hiệu gây hiểu nhầm không chỉ nằm trong văn bản. Một bài đăng có thể kết hợp tiêu đề giật gân, ảnh không khớp ngữ cảnh, nguồn dẫn có lịch sử không đáng tin, và quá trình lan truyền trong các cộng đồng có phong cách chính trị, châm biếm hoặc giải trí khác nhau. Vì vậy, các phương pháp hiện đại thường không chỉ dùng nội dung bài đăng mà còn khai thác cấu trúc quan hệ giữa bài đăng, người dùng, nguồn tin, cộng đồng và ảnh.

Mạng thông tin không đồng nhất là một biểu diễn phù hợp cho bài toán này. Trong một HIN, bài đăng không còn là một mẫu độc lập mà là một nút được nối với User, Subreddit, Domain và Image thông qua các quan hệ có ý nghĩa khác nhau. Heterogeneous graph neural networks như HeteroGraphSAGE có thể truyền thông tin theo từng loại quan hệ, nhờ đó học được biểu diễn giàu ngữ cảnh hơn so với mô hình chỉ dùng văn bản hoặc hình ảnh.

Tuy nhiên, ngữ cảnh quan hệ chỉ có ích khi nó bền vững giữa huấn luyện và triển khai. Nếu một loại nút hoặc quan hệ tương quan mạnh với nhãn trong tập huấn luyện nhưng không ổn định trong môi trường mới, mô hình có thể học lối tắt thay vì học cơ chế phát hiện tin giả. Trên Fakeddit, Subreddit là một confounder cấu trúc rất mạnh: trong mẫu nghiên cứu, 19 cộng đồng huấn luyện đều thuần nhãn. Điều này tạo ra đường tắt Post-Subreddit-Label gần như hoàn hảo. Một mô hình GNN có thể đạt điểm cao trên validation bằng cách nhận ra cộng đồng đăng tải, nhưng sụp giảm khi gặp cộng đồng mới hoặc khi tương quan cộng đồng-nhãn thay đổi.

Nghiên cứu này xem phát hiện tin giả trên HIN như một bài toán confounding cấu trúc. Giả thuyết chính là: HGNN huấn luyện theo ERM dễ khai thác shortcut cộng đồng; nếu chặn các đường truyền thông điệp qua confounder quan sát được, mô hình sẽ học biểu diễn bền hơn trong điều kiện dịch chuyển phân phối. Trên cơ sở đó, bài báo đề xuất CausalHeteroGNN, một kiến trúc hai nhánh gồm nhánh baseline trên đồ thị đầy đủ và nhánh nhân quả trên đồ thị đã can thiệp.

Các đóng góp chính của nghiên cứu gồm:

1. Đề xuất CausalHeteroGNN, một mô hình HeteroGraphSAGE đa phương thức với can thiệp cấu trúc ở mức lược đồ HIN nhằm loại bỏ đường truyền thông điệp qua Subreddit trong nhánh nhân quả.
2. Xây dựng Confounding-Shift Benchmark cho phát hiện tin giả trên HIN, trong đó tương quan confounder-nhãn bị đảo từ ρ=0.9 trong train/validation/seen test sang ρ=0.1 trong OOD test.
3. Đề xuất AutoCut, quy trình chọn quan hệ cần cắt bằng probe trên validation mà không dùng nhãn OOD.
4. Cung cấp phân tích rủi ro phương pháp luận, bao gồm rò rỉ FastRP, đóng góp của lịch sử domain, worst-group accuracy, LOCO và temporal shift.

## 2. Cơ sở lý thuyết và công trình liên quan

### 2.1. Phát hiện tin giả đa phương thức

Các mô hình phát hiện tin giả đa phương thức thường kết hợp văn bản, hình ảnh và mức độ nhất quán giữa hai kênh. Văn bản cung cấp nội dung tuyên bố trực tiếp, trong khi hình ảnh có thể bổ sung bằng chứng thị giác hoặc tiết lộ sự lệch ngữ cảnh. SAFE [12] khai thác ý tưởng này bằng cách đo tương đồng ngữ nghĩa giữa văn bản và ảnh. Các hướng khác sử dụng đồ thị lan truyền [11], tri thức ngoài [14] hoặc mạng không đồng nhất [4] để mô hình hóa bối cảnh xã hội của bài đăng.

Điểm còn thiếu trong nhiều nghiên cứu là đánh giá ngoài phân phối. Nếu train và test chia sẻ cùng cộng đồng, nguồn tin hoặc cơ chế nhãn, mô hình có thể dựa vào tương quan bề mặt. Khi đó, accuracy cao chưa chứng minh rằng mô hình hiểu nội dung tin giả. Bài báo này đặt trọng tâm vào câu hỏi: tín hiệu mà mô hình học được có còn hữu ích khi shortcut cộng đồng không còn đúng hay không?

### 2.2. Đồ thị không đồng nhất và HeteroGraphSAGE

Đồ thị không đồng nhất cho phép biểu diễn nhiều loại nút và quan hệ trong cùng một cấu trúc. Trong bài toán này, Post, User, Subreddit, Domain và Image có vai trò khác nhau; các quan hệ POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE và MEMBER_OF cũng mang ý nghĩa khác nhau. Nếu dùng GNN đồng nhất, thông tin loại quan hệ có thể bị làm mờ. HeteroGraphSAGE khắc phục điểm này bằng cách học hàm tổng hợp riêng theo từng loại quan hệ rồi kết hợp thông điệp tại nút đích.

Lợi thế của HeteroGraphSAGE là khả năng học quy nạp: mô hình có thể áp dụng cho nút mới nếu đặc trưng đầu vào và lược đồ quan hệ phù hợp. Tuy nhiên, trong đánh giá OOD, khả năng này chỉ có ý nghĩa nếu kiểm soát được rò rỉ từ test vào train. Vì vậy, các giao thức tự nhiên trong nghiên cứu này dùng chế độ inductive content-only: các cạnh chạm bài test được che khi cần thiết để tránh suy nhãn qua cấu trúc test.

### 2.3. Suy luận nhân quả và tổng quát hóa ngoài phân phối

Trong khung nhân quả [2,9], confounding xuất hiện khi một biến ảnh hưởng đồng thời đến biểu diễn đầu vào và nhãn quan sát được. Với Fakeddit, Subreddit ảnh hưởng đến phong cách và chủ đề nội dung, đồng thời trong dữ liệu huấn luyện lại gần như quyết định nhãn. Do đó, mô hình tối ưu ERM có thể học đường X <- C -> Y thay vì học quan hệ bền vững giữa nội dung và tính xác thực.

IRM [8] và EERM [13] cố gắng học biểu diễn bất biến thông qua các ràng buộc mềm theo môi trường. Tuy nhiên, khi tương quan confounder-nhãn quá mạnh, regularization mềm có thể không đủ để chống lại lợi ích phân loại tức thời từ shortcut. Cách tiếp cận của bài báo là can thiệp cứng ở mức cấu trúc: khi xác định Subreddit là confounder chính, nhánh nhân quả không được nhận thông điệp từ bất kỳ cạnh nào chạm Subreddit. Đây không phải ước lượng đầy đủ P(Y|X, do(C)), mà là một phẫu thuật trên đồ thị nhằm chặn đường truyền thông tin gây nhiễu.

## 3. Dữ liệu và biểu diễn HIN

### 3.1. Nguồn dữ liệu

Nghiên cứu sử dụng Fakeddit [1], bộ dữ liệu Reddit đa phương thức giai đoạn 2008-2020. Từ dữ liệu gốc, pipeline giữ lại 5,898 bài có ảnh thật để bảo đảm mỗi mẫu đều có cả kênh văn bản và hình ảnh. Split gốc gồm 5,000 bài train cân bằng 50/50, 400 bài validation và 498 bài test, trong đó 200 bài seen và 298 bài OOD. Hai cộng đồng r/neutralnews và r/theonion được loại khỏi train/validation để tạo test OOD ban đầu.

Việc chọn mẫu này phục vụ hai mục tiêu. Thứ nhất, quy mô dữ liệu đủ nhỏ để pipeline có thể tái lập trên phần cứng phổ thông. Thứ hai, dữ liệu bộc lộ rõ hiện tượng confounding cộng đồng: 19 Subreddit huấn luyện đều có fake-rate bằng 0 hoặc 1. Đây không phải nhiễu nhẹ mà là một shortcut gần như hoàn hảo.

### 3.2. Mô hình hóa đồ thị

Đồ thị thuộc tính gồm 17,079 nút và 28,274 quan hệ:

| Loại nút | Số lượng | Vai trò |
|---|---:|---|
| Post | 5,898 | Đối tượng cần phân loại |
| User | 4,604 | Người đăng hoặc tác nhân liên quan |
| Domain | 658 | Nguồn hoặc website được liên kết |
| Subreddit | 21 | Cộng đồng đăng tải |
| Image | 5,898 | Ảnh đi kèm bài đăng |

Năm loại quan hệ gồm POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE và MEMBER_OF. Neo4j được dùng để import, kiểm tra toàn vẹn bằng Cypher và chạy các thuật toán GDS phục vụ phân tích. Tuy nhiên, FastRP không được đưa vào input mặc định của mô hình vì có nguy cơ rò rỉ nhãn trong thiết lập OOD; nó chỉ được giữ cho phân tích leakage.

### 3.3. Đặc trưng đa phương thức

Mỗi Post có embedding tiêu đề từ SentenceTransformer all-mpnet-base-v2 với 768 chiều, ba đặc trưng số gồm score, upvote ratio và số bình luận, cùng đặc trưng **clip_cons** — cosine similarity giữa embedding CLIP của tiêu đề và embedding CLIP của ảnh (tổng cộng 772 chiều). Trên dữ liệu, clip_cons trung bình của bài thật là 0.295, còn bài giả là 0.245, gợi ý rằng tin giả có xu hướng lệch tiêu đề-ảnh mạnh hơn; vai trò của đặc trưng này được kiểm chứng bằng ablation ở Mục 6.2.

Ảnh được biểu diễn bằng CLIP ViT-B/32 với 512 chiều. User, Subreddit và Domain dùng các thống kê hành vi được tính riêng từ tập train; nút chưa thấy nhận giá trị trung tính 0.5. Quy tắc train-only rất quan trọng vì các đặc trưng lịch sử nếu tính trên toàn dữ liệu có thể chứa thông tin nhãn của test.

## 4. Phương pháp

### 4.1. SCM và biến gây nhiễu

Bài toán được mô tả bằng Mô hình Nhân quả Cấu trúc với các biến chính: nội dung văn bản X, ảnh I, nguồn tin D, người dùng U, cộng đồng C và nhãn Y. Trong đó, C = Subreddit là confounder cấu trúc chính. Subreddit ảnh hưởng đến phân phối nội dung vì mỗi cộng đồng có chủ đề và phong cách riêng; đồng thời trong dữ liệu huấn luyện, nó gần như quyết định nhãn. Nếu không kiểm soát, mô hình học quan hệ giữa cộng đồng và nhãn thay vì học dấu hiệu bền vững của tin giả.

User cũng có thể là confounder tiềm năng, nhưng cấu hình chính không cắt quan hệ Post-User. Lý do là AutoCut cho thấy cắt posted_by không loại được tín hiệu môi trường và còn làm mất thông tin hữu ích: probe vẫn đoán môi trường gần như hoàn hảo, trong khi OOD accuracy giảm mạnh. Vì vậy, nghiên cứu tập trung vào Subreddit như confounder mạnh nhất. Domain và User history vẫn còn trong nhánh nhân quả, và phần đóng góp của Domain history được đo riêng bằng ablation.

### 4.2. Kiến trúc CausalHeteroGNN

CausalHeteroGNN dùng một encoder HeteroGraphSAGE chung. Mỗi loại nút được chiếu về không gian ẩn d=96, đi qua ReLU và dropout 0.4, sau đó truyền thông điệp qua hai lớp SAGEConv theo từng loại quan hệ. Từ cùng encoder, mô hình chạy trên hai phiên bản đồ thị.

Nhánh baseline nhận đồ thị đầy đủ G. Nhánh này không phải baseline chính trong so sánh, mà là nhánh chẩn đoán: nó cho biết mô hình có thể khai thác tín hiệu giả mạnh đến đâu khi không bị chặn. Nhánh nhân quả nhận đồ thị đã can thiệp, trong đó mọi cạnh có nút nguồn hoặc nút đích thuộc loại Subreddit đều bị loại:

```text
E_causal = { (u, v) in E : type(u) != Subreddit and type(v) != Subreddit }.
```

Can thiệp này chặn toàn bộ đường truyền thông điệp trực tiếp qua Subreddit. Điểm quan trọng là thao tác được thực hiện ở mức lược đồ quan hệ của HIN, nơi loại nút và quan hệ có ý nghĩa ngữ nghĩa rõ ràng. Nhờ đó, mô hình loại bỏ shortcut cộng đồng một cách triệt để hơn so với việc chỉ phạt loss.

### 4.3. Phân rã biểu diễn và hàm mất mát

Ngoài cắt cạnh, mô hình dùng hai cơ chế phụ trợ. Thứ nhất là Gradient Reversal Layer với α=2.0: từ biểu diễn nhân quả, một bộ phân loại phụ cố đoán Subreddit-ID; gradient bị đảo để encoder học biểu diễn khó đoán cộng đồng hơn. Thứ hai là ràng buộc trực giao giữa biểu diễn baseline và biểu diễn nhân quả, nhằm giảm khả năng hai nhánh học trùng hoàn toàn cùng một tín hiệu.

Hàm mất mát tổng quát là:

```text
L = L_base,2w + 0.5 L_base,6w
  + L_causal,2w + 0.5 L_causal,6w
  + 0.5 L_spurious + 0.5 L_adv + 0.2 L_ortho.
```

Nhiệm vụ 2-way là phân loại thật/giả, còn nhiệm vụ 6-way giữ lại nhãn mịn của Fakeddit như một nhiệm vụ phụ. Mô hình được chọn bằng validation cùng phân phối huấn luyện và không dùng OOD để chọn checkpoint.

### 4.4. AutoCut

Một hạn chế tự nhiên của can thiệp cấu trúc là cần biết trước nên cắt quan hệ nào. AutoCut được đề xuất để giảm giả định này. Quy trình duyệt các ứng viên cắt quan hệ, bao gồm từng loại quan hệ đơn lẻ và cặp quan hệ {posted_in, member_of}. Với mỗi ứng viên, mô hình được huấn luyện ngắn; sau đó một probe tuyến tính cố đoán môi trường hoặc Subreddit từ biểu diễn nhân quả trên validation.

Ứng viên tốt là ứng viên làm probe khó đoán môi trường nhất nhưng vẫn giữ được khả năng phân loại. Tiêu chí chọn không dùng nhãn OOD, phù hợp với chuẩn training-domain validation trong domain generalization. Trên benchmark confounded, AutoCut chọn đúng cặp {posted_in, member_of} ở 3/3 seeds; trên dữ liệu tự nhiên không có confounder trội, lựa chọn dao động và khoảng cách probe nhỏ, cho thấy AutoCut không cắt mạnh khi bằng chứng không rõ.

## 5. Thiết kế thực nghiệm

### 5.1. Hệ đánh giá ba lớp

Bài báo không dùng một split duy nhất vì accuracy trên test thông thường không cho biết mô hình đang học nội dung hay học shortcut. Thay vào đó, hệ đánh giá gồm ba lớp, mỗi lớp trả lời một câu hỏi khác nhau.

Lớp thứ nhất là Confounding-Shift, dùng để đo cơ chế chính. Pipeline thay Subreddit thật bằng biến nhiễu nhị phân tổng hợp, tương quan với nhãn ở mức ρ=0.9 trong train/validation/seen test và đảo thành ρ=0.1 ở OOD test, theo tinh thần ColoredMNIST [8,10]. Nếu mô hình học shortcut, nó sẽ sụp khi tương quan đảo. Nếu mô hình học tín hiệu bền hơn, hiệu năng OOD sẽ giảm ít hơn. Đây là stress-test có kiểm soát, không nhằm mô phỏng đầy đủ thế giới thật.

Lớp thứ hai là can thiệp phản thực. Nội dung bài đăng được giữ nguyên, chỉ thay ngữ cảnh như do(Subreddit=neutral), do(Image=empty) hoặc do(Domain=credible). Chỉ số chính là Label-Flip Rate, tức tỷ lệ dự đoán đổi nhãn sau can thiệp. Lớp này kiểm tra mô hình có thực sự bất biến với Subreddit sau khi cắt hay không, đồng thời đo mức nhạy với ảnh và domain.

Lớp thứ ba là kiểm chứng tự nhiên bằng LOCO và temporal. LOCO giữ ra một hoặc một nhóm cộng đồng thật khỏi train rồi đưa vào OOD test. Temporal split huấn luyện trên 70% bài cũ nhất và đánh giá trên 10% bài mới nhất. Các giao thức này dùng chế độ inductive content-only khi cần, che các cạnh chạm bài test để tránh rò qua cấu trúc test.

### 5.2. Baseline

Các baseline được huấn luyện độc lập trên cùng dữ liệu và cùng cách chia:

| Baseline | Vai trò |
|---|---|
| ERM HeteroGraphSAGE | GNN thuần, tối ưu loss huấn luyện, không xử lý confounder |
| MLP content-only | Không dùng đồ thị; mốc neo cho tín hiệu nội dung |
| IRM | Phạt mềm để học biểu diễn bất biến theo môi trường cho trước |
| EERM | Sinh môi trường ảo bằng nhiễu cạnh, phạt bất biến trên đồ thị |

ERM, IRM và EERM dùng cùng backbone HeteroGraphSAGE với mô hình đề xuất, bao gồm số lớp, kích thước ẩn và siêu tham số chính. Do đó, khác biệt kết quả chủ yếu phản ánh cách xử lý confounder. MLP không dùng đồ thị và đóng vai trò mốc so sánh để xem phần lợi ích của GNN có vượt qua tín hiệu nội dung thuần hay không.

Các bảng chính báo cáo mean±std trên nhiều seed khi có thể. Kết quả chính của CausalHeteroGNN với đặc trưng nhất quán CLIP được chạy trên 5 seeds; một số baseline và ablation bổ sung được chạy trên 3 seeds và được ghi chú trực tiếp trong từng bảng. Khi chênh lệch nhỏ hơn dao động giữa folds hoặc seeds, bài chỉ kết luận là tương đương hoặc tiệm cận, không khẳng định thắng tuyệt đối.

## 6. Kết quả và thảo luận

### 6.1. Kết quả chính trên Confounding-Shift

**Bảng 1. Hiệu năng khi shortcut cộng đồng bị đảo trong Confounding-Shift.**
Kết quả chính cần đọc ở đây là khoảng cách giữa seen accuracy và OOD accuracy:
nếu mô hình dựa vào Subreddit, nó sẽ đạt điểm cao khi tương quan còn đúng nhưng
sụp khi tương quan bị đảo.

| Mô hình | Seen Acc% | OOD Acc% | AUC | F1-Drop% | Worst-Group% |
|---|---:|---:|---:|---:|---:|
| ERM | 90.6±0.9 | 55.0±4.0 | 0.551 | 43.0 | 21.0 |
| IRM † | 91.2±0.2 | 54.1±4.7 | 0.524 | 43.2 | 24.4 |
| EERM † | 93.2±0.2 | 59.6±1.1 | 0.694 | 38.6 | 28.6 |
| MLP content-only | 82.1±0.9 | 59.9±0.4 | 0.688 | 30.0 | 16.0 |
| **CausalHeteroGNN** | 84.2±0.9 | **79.9±4.2** | **0.922** | **5.4** | **56.0** |

*Ghi chú:* CausalHeteroGNN, ERM và MLP dùng bộ đặc trưng đầy đủ gồm `clip_cons`;
IRM/EERM (†) là đối chứng bổ sung để quan sát xu hướng của các regularizer bất
biến.

ERM đạt seen accuracy cao nhất trong nhóm GNN nhưng gần như mất khả năng phân biệt khi tương quan confounder-nhãn bị đảo. IRM và EERM cải thiện nhẹ nhưng vẫn không vượt rõ mốc content-only. Điều này ủng hộ giả thuyết rằng regularization mềm chưa đủ khi shortcut cộng đồng quá mạnh. CausalHeteroGNN thấp hơn ERM ở seen accuracy vì bỏ lối tắt, nhưng OOD accuracy cao hơn 24.9 điểm và F1-drop chỉ 5.4% so với 43.0%.

Worst-group accuracy cho thấy cùng một kết luận ở mức nhóm khó. Trong Confounding-Shift, nhóm khó nhất là nhóm có nhãn đi ngược thiên hướng môi trường, ví dụ bài giả trong môi trường thiên thật. CausalHeteroGNN là mô hình tốt nhất ở nhóm này. Riêng worst-group thấp của MLP không phản ánh shortcut cộng đồng, vì MLP không nhìn thấy môi trường, mà chủ yếu do calibration và ngưỡng quyết định lệch lớp.

### 6.2. Ablation đặc trưng nhất quán CLIP

Trong bản hoàn chỉnh, CausalHeteroGNN dùng `clip_cons` như một đặc trưng mặc định của Post. Để kiểm tra đặc trưng này có thật sự đóng góp hay chỉ làm tăng số chiều đầu vào, chúng tôi so sánh mô hình đầy đủ với ablation loại bỏ `clip_cons`.

| Chỉ số | Không dùng clip_cons | Mô hình đầy đủ | Thay đổi |
|---|---:|---:|---:|
| Conf OOD Acc | 74.2±3.6 | **79.9±4.2** | +5.7 |
| Conf AUC | 0.851 | **0.922** | +0.071 |
| Conf F1-Drop | 12.7 | **5.4** | -7.3 |
| Conf Worst-Group | 37.8 | **56.0** | +18.2 |
| Standard OOD Acc | 57.7±0.8 | **59.6±1.9** | +1.9 |
| Standard Worst-Group | 29.6 | **42.9** | +13.3 |

Điểm đáng chú ý là MLP và ERM không hưởng lợi tương tự từ cùng đặc trưng này. Điều đó gợi ý `clip_cons` không chỉ đơn giản là thêm một chiều feature; nó phát huy tác dụng khi được đặt trong nhánh đã giảm phụ thuộc vào Subreddit. Khi shortcut cộng đồng bị chặn, tín hiệu bất nhất văn bản-ảnh trở nên có giá trị hơn.

### 6.3. Can thiệp phản thực

**Bảng 2. Label-Flip Rate dưới các can thiệp.**

| Can thiệp | Baseline | Causal | Diễn giải |
|---|---:|---:|---|
| do(C1 = neutral) | 30.1% | ~0.0% | Kiểm chứng cơ chế cắt Subreddit |
| do(I = empty) | 4.0% | 8.0% | Causal nhạy hơn với tín hiệu ảnh |
| do(D = credible) | 6.2% | 15.5% | Causal dùng lịch sử nguồn rõ hơn |

LFR gần 0% dưới do(C1=neutral) là một sanity check theo thiết kế: vì nhánh nhân quả đã cắt toàn bộ cạnh chạm Subreddit, hoán đổi cộng đồng không còn làm đổi dự đoán. Hai can thiệp ảnh và domain có ý nghĩa thực nghiệm hơn. Khi bỏ ảnh hoặc thay đổi uy tín nguồn, nhánh nhân quả đổi dự đoán nhiều hơn, cho thấy sau khi bỏ shortcut cộng đồng, mô hình dựa nhiều hơn vào tín hiệu nội dung, ảnh và nguồn.

### 6.4. Ablation lịch sử nguồn tin

Một câu hỏi phản biện quan trọng là: nếu mô hình không dùng Subreddit, hiệu năng OOD đến từ đâu? Ablation `GNN_NEUTRAL_DOMAIN=1` đặt fake-ratio của Domain về 0.5, tức trung hòa lịch sử uy tín nguồn. Thí nghiệm này được chạy trong cùng điều kiện với ablation không dùng `clip_cons`, vì mục tiêu là đo phần chênh lệch tương đối do Domain history tạo ra.

| Cấu hình ablation | Conf OOD Acc% | OOD F1 | AUC |
|---|---:|---:|---:|
| CausalHeteroGNN, không trung hòa Domain | 74.2±3.6 | 0.731 | 0.851 |
| Domain fake-ratio = 0.5 | 58.8±2.9 | 0.534 | 0.681 |

Khoảng 15.4 điểm OOD đến từ lịch sử nguồn tin. Đây là tín hiệu hợp lệ nếu được tính train-only và nếu môi trường triển khai có domain tương đối ổn định. Tuy nhiên, nó cũng là một rủi ro OOD khi phân phối domain thay đổi. Vì vậy, kết luận cần được viết thận trọng: can thiệp cấu trúc loại bỏ confounder cộng đồng, nhưng hiệu năng tuyệt đối vẫn dựa một phần vào source credibility history.

### 6.5. Rò rỉ FastRP

FastRP tạo embedding cấu trúc toàn cục. Nếu tính FastRP trên toàn đồ thị trước khi chia train/test, vector của bài OOD đã chứa thông tin về cụm, cộng đồng và quan hệ với các nút khác. Trong dữ liệu có Subreddit thuần nhãn, thông tin này gần như mang nhãn test vào input.

Thực nghiệm leakage cho thấy khi giữ FastRP, OOD accuracy đạt khoảng 93.6-94.3% và AUC gần 0.98. Khi loại FastRP, OOD accuracy giảm về khoảng 61.0-61.1%. Chênh lệch hơn 33 điểm cho thấy kết quả cao là ảo. Vì vậy, FastRP được loại khỏi input mặc định và chỉ dùng cho phân tích rò rỉ hoặc dashboard.

### 6.6. AutoCut

Trên benchmark confounded, AutoCut chọn đúng cặp {posted_in, member_of} ở cả 3 seeds:

| Seed | Cut được chọn | Probe acc | OOD Acc |
|---|---|---:|---:|
| 42 | posted_in + member_of | 0.777 | 76.8 |
| 1 | posted_in + member_of | 0.755 | 82.9 |
| 2 | posted_in + member_of | 0.770 | 86.6 |

Các cut sai có probe accuracy cao hơn nhiều, khoảng 0.93-1.00, nghĩa là biểu diễn vẫn mang thông tin môi trường. Trên standard OOD tự nhiên, AutoCut không hội tụ vào một cut duy nhất: seed 42 chọn member_of, seed 1 chọn none, seed 2 chọn has_image. Khoảng cách probe nhỏ, nên kết luận phù hợp là không có confounder trội rõ như benchmark confounded. Đây là hành vi mong muốn: AutoCut nên cắt khi tín hiệu confounding rõ, và thận trọng khi dữ liệu không đủ bằng chứng.

### 6.7. LOCO và Temporal

LOCO được dùng như kiểm chứng thực tế, không phải bảng thắng chính. Confounding-Shift cho thấy cơ chế cắt confounder có hiệu quả khi shortcut bị đảo; LOCO kiểm tra xem shortcut cộng đồng có gây hại trên các cộng đồng thật chưa thấy hay không.

Mỗi fold LOCO giữ ra một cặp cộng đồng thật, gồm một cộng đồng thuần thật và một cộng đồng thuần giả, để tỷ lệ nhãn trong fold xấp xỉ 50/50. Điều này là cần thiết vì mọi cộng đồng trong mẫu đều thuần nhãn; nếu giữ ra một cộng đồng đơn lẻ, fold có thể chỉ chứa một nhãn.

**Bảng 3. LOCO 3 folds cộng đồng thật, seed 42, inductive, mô hình đầy đủ.**

| Fold giữ-ra | CausalHeteroGNN | ERM | MLP |
|---|---:|---:|---:|
| nottheonion + pareidolia | 62.3 | 57.9 | 61.3 |
| upliftingnews + fakehistoryporn | 70.5 | 56.8 | 73.1 |
| usnews+usanews + fakealbumcovers | 75.5 | 54.7 | 80.5 |
| **Trung bình Acc / Macro-F1** | **69.4±5.4 / 0.677** | **56.5±1.3 / 0.403** | **71.6±7.9 / 0.706** |

Cách đọc đúng của bảng này là: CausalHeteroGNN vượt rõ ERM GNN nhưng gần tương đương MLP content-only. Chênh lệch giữa Causal và MLP chỉ 2.2 điểm, nhỏ hơn dao động giữa folds. Vì vậy, LOCO không chứng minh CausalHeteroGNN luôn tốt hơn content-only. Giá trị của LOCO là cho thấy shortcut cộng đồng có thật trên dữ liệu tự nhiên, và can thiệp cấu trúc giúp GNN tránh bị cấu trúc cộng đồng kéo sai.

Temporal split có OOD là 10% bài mới nhất, fake-rate chỉ 0.23. Accuracy dao động mạnh giữa mô hình, nhưng AUC của Causal, ERM và MLP gần nhau: lần lượt 0.662, 0.662 và 0.688. Điều này gợi ý dịch chuyển thời gian trong Fakeddit chủ yếu là prior shift, tức tỷ lệ nhãn thay đổi theo thời gian. Bài báo vì vậy không tuyên bố CausalHeteroGNN giải quyết temporal shift; đây là một giới hạn và là hướng cho calibration theo thời gian.

### 6.8. Kết quả âm tính và giới hạn

Nghiên cứu ghi nhận một số kết quả âm tính. GroupDRO không cải thiện so với ERM trong thiết lập này. Cổng gradient để học phép cắt thất bại vì encoder hấp thụ áp lực đối kháng trong khi gates không đóng đủ. Tiêu chí IRM-penalty cũng có thể chọn sai vì bỏ sót đường rò hai bước qua member_of.

Các giới hạn chính gồm: nhận diện satire, đặc biệt nhóm theonion|Fake, vẫn khó; temporal prior shift chưa được xử lý; phạm vi thực nghiệm mới trên một bộ dữ liệu; và một phần hiệu năng dựa vào lịch sử domain nên có thể giảm khi nguồn tin thay đổi mạnh. Những giới hạn này không phủ nhận kết quả chính, nhưng xác định rõ phạm vi mà kết luận của bài báo có thể áp dụng.

## 7. Kết luận

Nghiên cứu cho thấy các mô hình đồ thị không đồng nhất cho phát hiện tin giả đa phương thức có thể thất bại do shortcut cấu trúc. Trên Fakeddit, Subreddit là một confounder rất mạnh: nó giúp mô hình đạt accuracy cao trong phân phối đã thấy, nhưng làm mô hình gần như ngẫu nhiên khi tương quan cộng đồng-nhãn bị đảo.

CausalHeteroGNN giải quyết thất bại này bằng can thiệp cấu trúc. Thay vì chỉ thêm regularization mềm, mô hình loại bỏ toàn bộ đường truyền thông điệp qua Subreddit trong nhánh nhân quả, đồng thời dùng adversarial và orthogonality losses để giảm tín hiệu cộng đồng còn sót. Kết quả trên Confounding-Shift, counterfactual intervention và LOCO ủng hộ luận điểm rằng chặn đường gây nhiễu trong HIN giúp cải thiện tổng quát hóa ngoài phân phối so với huấn luyện HGNN thông thường.

Đồng thời, nghiên cứu cũng chỉ ra những rủi ro cần kiểm soát khi đánh giá GNN trên dữ liệu tin giả: FastRP có thể gây rò rỉ nhãn nghiêm trọng, lịch sử domain đóng góp đáng kể vào hiệu năng, satire vẫn khó nhận diện, và temporal prior shift chưa được giải quyết. Hướng phát triển tiếp theo gồm mở rộng sang nhiều bộ dữ liệu, kiểm tra dịch chuyển nguồn tin, cải thiện nhận diện satire bằng tín hiệu ngữ dụng, hiệu chỉnh calibration theo thời gian và mở rộng AutoCut cho các lược đồ HIN phức tạp hơn.

## Phụ lục A. Tái lập và sản phẩm nghiên cứu

Pipeline cuối cùng nằm trong thư mục `pipeline/`, gồm 17 script chạy theo 7 phase: chuẩn bị dữ liệu và đồ thị; huấn luyện CausalHeteroGNN; huấn luyện baseline; AutoCut; worst-group và bảng tổng hợp; sinh hình; và dashboard Streamlit. Kết quả cuối được gom trong `results/final_tables.md` và đối chiếu với `STATS_MASTER.md`.

Sản phẩm của đề tài gồm HIN Fakeddit trong Neo4j, các split chuẩn gồm original, Confounding-Shift, temporal và LOCO, kết quả JSON trace, bảng tổng hợp, dashboard trực quan hóa, và các tài liệu phản biện phương pháp luận. Khi nộp hội nghị có giới hạn trang, phụ lục này nên được rút gọn thành mô tả mã nguồn và liên kết tái lập.

## Tài liệu tham khảo

[1] K. Nakamura, S. Levy, W. Y. Wang, "r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection," LREC 2020.

[2] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., Cambridge University Press, 2009.

[3] W. L. Hamilton, R. Ying, J. Leskovec, "Inductive representation learning on large graphs," NeurIPS 2017.

[4] Z. Hu, Y. Dong, K. Wang, Y. Sun, "Heterogeneous graph transformer," WWW 2020.

[5] Y. Ganin et al., "Domain-adversarial training of neural networks," JMLR 17(59), 2016.

[6] B. Scholkopf, "Causality for machine learning," in *Probabilistic and Causal Inference: The Works of Judea Pearl*, ACM, 2022.

[7] R. Ying et al., "GNNExplainer: Generating explanations for graph neural networks," NeurIPS 2019.

[8] M. Arjovsky, L. Bottou, I. Gulrajani, D. Lopez-Paz, "Invariant risk minimization," arXiv:1907.02893, 2019.

[9] J. Peters, D. Janzing, B. Scholkopf, *Elements of Causal Inference*, MIT Press, 2017.

[10] I. Gulrajani, D. Lopez-Paz, "In search of lost domain generalization," ICLR 2021.

[11] T. Bian et al., "Rumor detection on social media with bi-directional graph convolutional networks," AAAI 2020.

[12] X. Zhou, J. Wu, R. Zafarani, "SAFE: Similarity-aware multi-modal fake news detection," PAKDD 2020.

[13] Q. Wu, H. Zhang, J. Yan, D. Wipf, "Handling distribution shifts on graphs: An invariance perspective," ICLR 2022.

[14] Z. Liu, C. Xiong, M. Sun, Z. Liu, "Fine-grained fact verification with kernel graph attention network," ACL 2020.

[15] I. Robinson, J. Webber, E. Eifrem, *Graph Databases*, 2nd ed., O'Reilly, 2015.

[16] Y. Wu et al., "Discovering invariant rationales for graph neural networks," ICLR 2022.

[17] Y. Chen et al., "Learning causally invariant representations for out-of-distribution generalization on graphs," NeurIPS 2022.
