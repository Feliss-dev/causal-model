Causal Graph Disentanglement with Heterogeneous GraphSAGE for Multimodal Misinformation Detection
Phân rã quan hệ nhân quả trên đồ thị hỗn hợp bằng GraphSAGE cho phát hiện tin giả đa phương thức

Abstract - Tóm tắt. Mạng nơ-ron đồ thị hỗn hợp (Heterogeneous Graph Neural Networks - HGNNs) là một hướng tiếp cận hiệu quả cho bài toán phát hiện tin giả đa phương thức nhờ khả năng mô hình hóa đồng thời nhiều loại thực thể, quan hệ và tín hiệu nội dung. Tuy nhiên, trong các môi trường mạng xã hội, hiệu năng cao trên tập kiểm thử cùng phân phối không đồng nghĩa với khả năng tổng quát hóa. Các mô hình đồ thị có thể khai thác các tương quan giả (spurious correlations), đặc biệt là lịch sử phân bố nhãn của cộng đồng đăng tải (Subreddit), thay vì học các tín hiệu nội dung ổn định. Nghiên cứu này đề xuất CausalHeteroGNN, một kiến trúc Heterogeneous GraphSAGE đầy đủ tín hiệu text-image, dựa trên mô hình nhân quả cấu trúc (SCM), trong đó nhánh nhân quả thực hiện can thiệp cấu trúc bằng cách loại bỏ các đường truyền thông tin liên quan đến Subreddit. Trên Confounding-Shift Benchmark, CausalHeteroGNN đạt 79.9±4.2% OOD Accuracy, AUC 0.922 và F1-drop chỉ 5.4%, vượt Baseline GNN (ERM HeteroSAGE, 52.1±7.8%), IRM (54.1±4.7%), MLP content-only (59.5±1.7%) và EERM (59.6±1.1%). Trên các phép chia cộng đồng tự nhiên LOCO, CausalHeteroGNN đạt 69.4±5.4%, cao hơn rõ rệt Baseline GNN (56.5±1.3%) và gần với MLP content-only (71.6±7.9%). Nghiên cứu đồng thời chỉ ra nguy cơ rò rỉ nhãn từ FastRP khi đặc trưng cấu trúc được tính trên toàn bộ đồ thị trước khi tách train-test, có thể làm tăng ảo OOD Accuracy hơn 33 điểm phần trăm. Kết quả cho thấy can thiệp cấu trúc trực tiếp kết hợp tín hiệu nhất quán văn bản-hình ảnh là một hướng khả thi để kiểm soát shortcut cộng đồng trong phát hiện tin giả đa phương thức, đồng thời nhấn mạnh nhu cầu đánh giá OOD bằng nhiều tầng kiểm tra thay vì chỉ dựa vào một split chuẩn.

Keywords: HGNNs, SCM, Multimodal Misinformation Detection, GraphSAGE, OOD Generalization, Causal Intervention

1. Introduction - Giới thiệu

Sự phát triển của truyền thông đa phương tiện làm cho tin giả không còn chỉ tồn tại dưới dạng văn bản, mà thường kết hợp tiêu đề, hình ảnh, nguồn đăng tải, lịch sử cộng đồng và các tín hiệu tương tác xã hội. Điều này tạo ra nhu cầu đối với các mô hình có khả năng kết hợp nhiều loại thông tin trong cùng một không gian học. Các mạng nơ-ron đồ thị hỗn hợp (HGNNs) đáp ứng tốt yêu cầu này vì có thể biểu diễn đồng thời các nút Post, User, Subreddit, Domain và Image cùng các quan hệ giữa chúng.

Tuy nhiên, chính khả năng khai thác cấu trúc đồ thị cũng làm HGNN dễ học các đường đi tắt. Trong các tập dữ liệu như Fakeddit, nhiều cộng đồng Reddit có phân bố nhãn rất lệch; một số cộng đồng gần như toàn tin thật, trong khi một số khác gần như toàn tin giả. Khi mô hình được phép truyền thông tin qua Subreddit, nó có thể dự đoán nhãn dựa trên lịch sử cộng đồng thay vì nội dung của bài đăng. Kết quả là mô hình có thể đạt độ chính xác cao trong môi trường quen thuộc nhưng suy giảm mạnh khi phân phối cộng đồng thay đổi.

Từ góc nhìn nhân quả, Subreddit đóng vai trò như một biến gây nhiễu: nó ảnh hưởng đến cả biểu diễn quan sát được của bài đăng và nhãn dự đoán, tạo ra đường đi cửa sau giữa đặc trưng và nhãn. Vì vậy, câu hỏi trung tâm của nghiên cứu không chỉ là mô hình đạt Accuracy bao nhiêu, mà là liệu mô hình có còn dự đoán ổn định khi tín hiệu cộng đồng bị giữ lại, loại bỏ, hoán đổi hoặc đảo chiều hay không.

Nghiên cứu này đưa ra bốn đóng góp chính:

1. Đề xuất CausalHeteroGNN, một kiến trúc Heterogeneous GraphSAGE có nhánh nhân quả thực hiện can thiệp cấu trúc bằng cách loại bỏ các cạnh liên quan đến Subreddit, nhằm kiểm soát shortcut cộng đồng.
2. Xây dựng Confounding-Shift Benchmark, một stress-test có kiểm soát trong đó tương quan giữa biến gây nhiễu và nhãn bị đảo từ rho=0.9 ở train/seen sang rho=0.1 ở OOD.
3. Bổ sung đánh giá tự nhiên bằng LOCO (Leave-One-Community-Out) và Temporal split để phân biệt hiệu quả dưới dịch chuyển cộng đồng thật với giới hạn dưới dịch chuyển thời gian/prior shift.
4. Phát hiện và lượng hóa rò rỉ nhãn từ FastRP trong thiết lập quy nạp, từ đó đề xuất quy trình đánh giá không rò rỉ cho mô hình đồ thị đa phương thức.

2. Background and Related Works - Cơ sở lý thuyết và các công trình liên quan

Các nghiên cứu về phát hiện tin giả đa phương thức cho thấy việc kết hợp văn bản, hình ảnh và ngữ cảnh xã hội giúp cải thiện khả năng nhận diện thông tin sai lệch. SAFE [12] khai thác tính nhất quán ngữ nghĩa giữa văn bản và hình ảnh; BiGCN [11] mô hình hóa quá trình lan truyền thông tin; KGAT [14] tích hợp tri thức ngoài; trong khi các kiến trúc HGNN như HGT [4] cho phép biểu diễn nhiều loại thực thể và quan hệ trong cùng một đồ thị. Fakeddit [1] là một benchmark lớn, cung cấp văn bản, hình ảnh, metadata và nhãn ở nhiều mức độ, nên phù hợp để nghiên cứu các mô hình đa phương thức.

Xu hướng gần đây của lĩnh vực không chỉ tập trung vào tăng Accuracy trong cùng phân phối, mà còn chú ý nhiều hơn đến khả năng tổng quát hóa ngoài phân phối (OOD), tính bền vững trước domain mới và khả năng giải thích. Các khảo sát gần đây về phát hiện tin giả đa phương thức nhấn mạnh vai trò ngày càng tăng của Transformer, GNN, social context modeling, external knowledge và LLM reasoning, đồng thời chỉ ra rằng generalization vẫn là một thách thức mở [16]. Các benchmark mới như MMFakeBench [17] cũng cho thấy các mô hình hiện tại còn gặp khó khăn trong bối cảnh sai lệch đa nguồn và đa kiểu giả mạo.

Từ phía học đồ thị OOD, các phương pháp invariant learning, causal graph learning và subgraph disentanglement đang được phát triển mạnh. IRM [8] tìm kiếm biểu diễn bất biến giữa nhiều môi trường; EERM [13] tạo môi trường ảo bằng nhiễu loạn đồ thị để giảm phụ thuộc vào môi trường cụ thể. Gần đây, các mô hình như CSDA [18] trích xuất cấu trúc con bất biến để xử lý phát hiện tin giả cross-domain. Tuy nhiên, nhiều phương pháp vẫn dựa vào ràng buộc mềm hoặc mask học được, nên khó đảm bảo rằng đường truyền thông tin gây nhiễu đã bị loại bỏ hoàn toàn.

Nghiên cứu này chọn một hướng trực tiếp hơn: nếu Subreddit là biến gây nhiễu đã được xác định rõ trong SCM, nhánh nhân quả sẽ cắt các quan hệ liên quan đến Subreddit ở cấp độ cấu trúc. Cách tiếp cận này không nhằm thay thế toàn bộ các phương pháp invariant learning, mà đóng vai trò như một phép can thiệp nhân quả rõ ràng, có thể kiểm chứng bằng counterfactual swap và worst-group analysis.

3. Data Collection and Representation in Neo4j

Nghiên cứu sử dụng Fakeddit [1], một benchmark thu thập từ Reddit giai đoạn 2008-2020 với hơn một triệu mẫu gốc, bao gồm văn bản, hình ảnh, metadata và nhãn 2-way/3-way/6-way. Do giới hạn hạ tầng, thí nghiệm trong nghiên cứu này được thực hiện trên 5,898 bài đăng có ảnh thật. Sau khi mô hình hóa dưới dạng đồ thị thuộc tính, dữ liệu gồm 17,079 nút và 28,274 quan hệ.

Đồ thị được lưu trữ và kiểm tra bằng Neo4j [15], gồm năm loại nút: Post, User, Subreddit, Domain và Image; cùng năm loại quan hệ có hướng: POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE và MEMBER_OF. Cụ thể, đồ thị có 5,898 nút Post, 4,604 nút User, 658 nút Domain, 21 nút Subreddit và 5,898 nút Image.

Split gốc gồm 5,000 bài train (2,500 real và 2,500 fake), 400 bài validation và 498 bài test. Trong test có 200 mẫu seen và 298 mẫu OOD từ hai cộng đồng r/neutralnews và r/theonion. Tập validation chỉ dùng để chọn siêu tham số; tập test được giữ nguyên cho đánh giá cuối cùng.

Mỗi bài đăng được chuyển thành đối tượng HeteroData của PyTorch Geometric. Đặc trưng Post mặc định trong mô hình đề xuất gồm embedding tiêu đề từ all-mpnet-base-v2 (768 chiều), ba thuộc tính thống kê (score, upvote ratio, số bình luận) và CLIPcons, tức cosine similarity giữa embedding văn bản và hình ảnh từ CLIP ViT-B/32. Do đó, Post feature của mô hình đầy đủ có 772 chiều. Các nút User, Subreddit và Domain được mô tả bằng các thống kê hành vi tính từ tập huấn luyện. FastRP không được dùng trong input mặc định của mô hình chính; nó chỉ được giữ lại cho thí nghiệm chẩn đoán rò rỉ ở Mục 5.6.

Một điểm quan trọng của dữ liệu là 19 cộng đồng train đều có fake-rate cực đoan, bằng 0.0 hoặc 1.0. Điều này làm shortcut cộng đồng trở thành một hiện tượng tự nhiên trong dữ liệu, không chỉ là giả định nhân tạo. Confounding-Shift Benchmark được xây dựng để khuếch đại và kiểm soát hiện tượng này, còn LOCO được dùng để kiểm tra trên các cộng đồng tự nhiên chưa thấy.

4. Proposed Graph Neural Network with GraphSAGE

CausalHeteroGNN được thiết kế theo kiến trúc hai nhánh. Nhánh phụ nhận đồ thị gốc G với đầy đủ quan hệ để hấp thụ tín hiệu gây nhiễu, trong khi nhánh causal nhận đồ thị đã can thiệp G_causal, trong đó các cạnh liên quan đến Subreddit bị loại bỏ. Hai nhánh cùng dựa trên Heterogeneous GraphSAGE nhưng được tối ưu với mục tiêu tách biểu diễn ổn định khỏi biểu diễn gây nhiễu.

Hình 1. Mô hình biểu diễn dữ liệu Fakeddit dưới dạng đồ thị trong Neo4j.

Hình 2. Kiến trúc CausalHeteroGNN: can thiệp cấu trúc loại bỏ cạnh Subreddit để tạo đồ thị nhân quả G_causal; nhánh causal và nhánh phụ trên đồ thị gốc được mã hóa song song, tách rời bằng Gradient Reversal Layer (GRL, alpha=2.0) và ràng buộc trực giao.

Do các loại nút có không gian đặc trưng khác nhau, mỗi loại nút được ánh xạ vào không gian ẩn chung d=96 bằng phép biến đổi tuyến tính độc lập, theo sau là ReLU và dropout p=0.4. Hai lớp Heterogeneous GraphSAGE thực hiện lan truyền thông tin theo từng loại quan hệ. Với mỗi quan hệ r, biểu diễn mới của một nút được cập nhật từ trạng thái hiện tại và trung bình biểu diễn của các láng giềng tương ứng.

Can thiệp nhân quả được định nghĩa như sau:

E_causal = { (u,v) in E : tau(u) != Subreddit and tau(v) != Subreddit }

Thao tác này cắt các đường truyền thông tin liên quan đến biến gây nhiễu cộng đồng. Trong cách diễn giải SCM, đây là một can thiệp cấu trúc nhằm khóa đường đi cửa sau từ cộng đồng đến nhãn. Lưu ý rằng nghiên cứu không cắt Domain và User history trong cấu hình chính; đây là các tín hiệu có thể hữu ích trong triển khai thực tế nhưng cần được kiểm tra thêm trong các phân tích chuyên biệt.

Hàm mất mát đa mục tiêu gồm các thành phần phân loại nhị phân và đa lớp ở hai nhánh, tổn thất spurious, tổn thất đối kháng từ GRL và ràng buộc trực giao:

L = L_base,2w + 0.5 L_base,6w + L_causal,2w + 0.5 L_causal,6w + 0.5 L_spurious + 0.5 L_adv + 0.2 L_ortho

Trong đó L_base và L_causal là tổn thất phân loại của hai nhánh; L_spurious khuyến khích nhánh phụ hấp thụ tín hiệu gây nhiễu; L_adv làm giảm khả năng mã hóa môi trường trong biểu diễn causal; L_ortho khuyến khích hai không gian biểu diễn tách biệt.

5. Experiments - Thực nghiệm

5.1. Giao thức đánh giá

Nghiên cứu sử dụng ba tầng đánh giá để tránh phụ thuộc vào một benchmark duy nhất:

Tầng 1 - Đánh giá tự nhiên. Held-Out Subreddit giữ lại r/neutralnews và r/theonion cho kiểm thử; LOCO giữ ra các cặp cộng đồng mới; Temporal split dùng 70% bài cũ nhất để train và 10% bài mới nhất làm OOD. Đây là các phép chia không đảo nhãn nhân tạo.

Tầng 2 - Can thiệp counterfactual. Các phép do(.) thay đổi cộng đồng, hình ảnh hoặc độ tin cậy domain trong khi giữ nguyên nội dung chính, nhằm đo mức phụ thuộc của mô hình vào từng nhóm tín hiệu.

Tầng 3 - Stress-test có kiểm soát. Confounding-Shift thay Subreddit thực bằng biến gây nhiễu nhị phân tổng hợp (spur_fakebias, spur_realbias). Pha train/seen dùng rho=0.9, tức biến gây nhiễu tương quan mạnh với nhãn; pha OOD dùng rho=0.1, tức tương quan bị đảo. Giao thức này tương tự tinh thần ColoredMNIST, không đại diện trực tiếp cho thế giới thật nhưng hữu ích để khuếch đại shortcut và kiểm tra cơ chế.

5.2. Mô hình đối chứng và tiêu chí

CausalHeteroGNN được so sánh với các đối chứng sau:

- MLP content-only: chỉ dùng đặc trưng nội dung của Post, đóng vai trò ước lượng trần content khi không dùng cấu trúc đồ thị.
- Baseline GNN (ERM HeteroSAGE): mô hình gốc, huấn luyện bằng empirical risk minimization trên đồ thị đầy đủ, không có cơ chế xử lý nhiễu.
- IRM: baseline mở rộng từ mô hình gốc, bổ sung ràng buộc học biểu diễn bất biến giữa các môi trường, lambda=100.
- EERM: baseline mở rộng từ mô hình gốc, tạo 3 môi trường ảo bằng nhiễu loạn cạnh với xác suất tối đa 0.3 để học dưới dịch chuyển đồ thị.

Các chỉ số gồm Accuracy, Macro-F1, AUC, F1-drop, Worst-Group Accuracy và Label-Flip Rate (LFR). Các baseline chính được tính trung bình trên 3 seeds {42,1,2}; mô hình đề xuất CausalHeteroGNN được đánh giá trên 5 seeds {42,1,2,3,4}. Nhánh phụ nội bộ trong CausalHeteroGNN chỉ được dùng như chẩn đoán shortcut, không dùng làm baseline chính.

5.3. Kết quả chính trên Standard OOD và Confounding-Shift

Trên Held-Out Subreddit OOD, các mô hình đạt hiệu năng khá gần nhau. Điều này cho thấy split held-out một fold chưa đủ áp lực để phân biệt rõ mô hình học nội dung ổn định với mô hình khai thác shortcut cộng đồng. MLP content-only đạt 60.5±1.1%; EERM đạt 60.9±1.9%; Baseline GNN đạt 57.6±2.7%; CausalHeteroGNN đạt 59.6±1.9%, gần trần content của MLP có CLIPcons (60.3±0.4%).

Bảng 1. Hiệu năng OOD chính trên hai giao thức, mean±std.

| Mô hình | Seen Acc | Held-Out OOD | Conf-Shift OOD | AUC (Conf) | F1-drop (Conf) |
|---|---:|---:|---:|---:|---:|
| MLP content-only | 81.7±0.6 | 60.5±1.1 | 59.5±1.7 | 0.685 | 30.0% |
| Baseline GNN (ERM HeteroSAGE) | 91.0±0.4 | 57.6±2.7 | 52.1±7.8 | 0.511 | 45.1% |
| IRM | 91.2±0.2 | 56.9±3.0 | 54.1±4.7 | 0.524 | 43.2% |
| EERM | 93.2±0.2 | 60.9±1.9 | 59.6±1.1 | 0.694 | 38.6% |
| CausalHeteroGNN† | 84.2±0.9 | 59.6±1.9 | 79.9±4.2 | 0.922 | 5.4% |

*†CausalHeteroGNN là mô hình đầy đủ với CLIPcons (tín hiệu nhất quán văn bản-hình ảnh), trung bình trên 5 seeds {42,1,2,3,4}. Các baseline (MLP, Baseline GNN, IRM, EERM) không có CLIPcons, trung bình trên 3 seeds {42,1,2}. AUC báo cáo trên tập Confounding-Shift OOD.*

Trên Confounding-Shift, khác biệt trở nên rõ rệt. Baseline GNN đạt 52.1±7.8% và AUC 0.511, gần mức ngẫu nhiên, cho thấy mô hình gốc mất khả năng phân biệt khi tương quan cộng đồng-nhãn bị đảo. IRM chỉ cải thiện nhẹ lên 54.1±4.7%; EERM và MLP đạt xấp xỉ 59.5-59.6%. CausalHeteroGNN đạt 79.9±4.2%, cao hơn Baseline GNN 27.8 điểm phần trăm và cao hơn EERM 20.3 điểm phần trăm. Đồng thời, F1-drop của CausalHeteroGNN chỉ 5.4%, thấp hơn nhiều so với Baseline GNN (45.1%) và EERM (38.6%).

Nhánh chẩn đoán nội bộ trong mô hình đạt 36.4±8.1% và AUC 0.314. Kết quả này không dùng làm baseline chính, nhưng có giá trị giải thích: nhánh được khuyến khích hấp thụ tín hiệu spurious có ranh giới quyết định bị đảo mạnh khi môi trường thay đổi.

Hình 3. So sánh tất cả mô hình trên Confounding-Shift OOD: Baseline GNN sụp đổ về mức ngẫu nhiên khi tương quan cộng đồng-nhãn bị đảo; CausalHeteroGNN+CLIPcons duy trì 79.9%.

Hình 4. So sánh hiệu năng OOD giữa hai giao thức: Held-Out Subreddit (các mô hình gần nhau ~57–61%) và Confounding-Shift (phân biệt rõ nhờ can thiệp cấu trúc, CausalHeteroGNN+CLIPcons đạt 79.9% so với 52.1% của Baseline GNN).

5.4. Worst-Group Accuracy

Accuracy trung bình có thể che khuất các nhóm khó, đặc biệt trong OOD nơi một số nhóm bị đảo tương quan mạnh hơn các nhóm khác. Vì vậy, nghiên cứu tính Worst-Group Accuracy theo nhóm env x label.

Bảng 3. Worst-Group Accuracy trên Confounding-Shift (mean±std, 3 seeds).

| Mô hình | Worst-Group Acc | Avg-Group Acc |
|---|---:|---:|
| MLP content-only | 15.6±3.1 | 56.5±1.0 |
| Baseline GNN | 23.5±7.9 | 73.2±4.4 |
| IRM | 24.4±7.1 | 74.3±2.7 |
| EERM | 28.6±1.3 | 77.0±1.0 |
| CausalHeteroGNN | 37.8±11.3 | 70.8±4.5 |

Các phương pháp soft-penalty như IRM và EERM vẫn suy giảm ở nhóm khó nhất, đặc biệt là các bài Fake trong môi trường thiên Real. CausalHeteroGNN đạt worst-group 37.8±11.3%, cao hơn tất cả các phương pháp so sánh, cho thấy can thiệp cấu trúc cải thiện không chỉ ở Accuracy trung bình mà còn ở nhóm dễ bị tổn thương nhất. Lưu ý rằng CausalHeteroGNN có avg-group 70.8%, cao hơn Baseline GNN (73.2%) ít hơn về mặt trung bình nhưng tốt hơn nhiều ở worst-group; điều này phản ánh sự đánh đổi giữa tối ưu trung bình và tính bền vững nhóm cực trị.

5.5. Đánh giá trên chia tách tự nhiên: LOCO và Temporal

Để trả lời phản biện rằng Confounding-Shift là benchmark nhân tạo, nghiên cứu bổ sung LOCO (Leave-One-Community-Out) trên các cặp cộng đồng tự nhiên chưa thấy trong train. Các split này không đảo nhãn nhân tạo; toàn bộ thống kê node được tính lại từ train của split mới để tránh rò rỉ. Trong lần bổ sung này, IRM và EERM cũng được đánh giá trên cùng ba fold để hoàn thiện bảng so sánh.

Bảng 4. LOCO trên ba fold tự nhiên. Accuracy (%) trên fold chưa thấy. MLP/Baseline/CausalHeteroGNN: single seed=42. IRM/EERM: mean±std trên 3 seeds {42,1,2}.

| Fold held-out | MLP† | Baseline GNN† | IRM‡ | EERM‡ | CausalHeteroGNN† |
|---|---:|---:|---:|---:|---:|
| nottheonion + pareidolia | 61.3 | 57.9 | 62.0±4.3 | 62.2±0.1 | 62.3 |
| upliftingnews + fakehistoryporn | 73.1 | 56.8 | 66.5±4.0 | 70.7±1.0 | 70.5 |
| usnews+usanews + fakealbumcovers | 80.5 | 54.7 | 69.1±4.8 | 76.9±2.9 | 75.5 |
| **Mean±std (across folds)** | **71.6±7.9** | **56.5±1.3** | **65.8±2.9** | **70.0±6.0** | **69.4±5.4** |

†single seed=42. ‡mean±std trên 3 seeds {42,1,2}.

Kết quả LOCO cho thấy shortcut cộng đồng không chỉ là hiện tượng synthetic. Baseline GNN suy giảm ổn định trên mọi fold với trung bình 56.5±1.3%, trong khi các mô hình khác dao động đáng kể theo từng fold. IRM đạt trung bình 65.8±2.9%, cao hơn Baseline GNN nhưng thấp hơn CausalHeteroGNN (69.4±5.4%). EERM đạt 70.0±6.0%, tương đương CausalHeteroGNN về trung bình nhưng dao động lớn hơn nhiều (std 6.0 điểm, so với 5.4 điểm). Fold loco_a (nottheonion+pareidolia) là fold khó nhất; tại đây IRM đạt 62.0±4.3% và EERM đạt 62.2±0.1%, tương đương CausalHeteroGNN (62.3%), cho thấy trên cộng đồng cụ thể này regularizer bất biến cũng cạnh tranh được.

Điều này gợi ý rằng can thiệp cấu trúc ổn định hơn về hành vi trên nhiều cộng đồng khác nhau — std của CausalHeteroGNN (5.4) thấp hơn EERM (6.0) và IRM (2.9 giữa các fold nhưng 4-5 điểm trong fold). Trên cộng đồng hoàn toàn mới, cấu trúc đồ thị không luôn thêm thông tin so với nội dung; giá trị của CausalHeteroGNN là giữ được lợi ích đồ thị mà không để shortcut cộng đồng làm hỏng dự đoán như Baseline GNN.

Hình 9. So sánh tất cả mô hình trên ba fold LOCO và trung bình (†seed=42; ‡mean 3 seeds).

Temporal split cho kết quả khác. Khi train trên 70% bài cũ nhất và kiểm thử trên 10% bài mới nhất, tập OOD có fake-rate chỉ 0.23, tạo ra label-prior shift mạnh. Accuracy và Macro-F1 dao động lớn giữa các mô hình, nhưng AUC gần nhau.

Bảng 5. Temporal split, OOD là 10% bài mới nhất, 3 seeds.

| Mô hình | OOD Acc | Macro-F1 | AUC |
|---|---:|---:|---:|
| CausalHeteroGNN | 49.5±19.1 | 0.449 | 0.662 |
| Baseline GNN | 79.5±1.0 | 0.614 | 0.662 |
| MLP | 23.1±0.6 | 0.191 | 0.688 |

Cách đọc phù hợp là không xem Baseline GNN vượt trội thật sự trong Temporal, vì AUC của ba mô hình gần tương đương. Chênh lệch Accuracy chủ yếu đến từ ngưỡng quyết định dưới prior shift: Baseline GNN thiên về majority-Real nên có Accuracy cao khi OOD có ít Fake; MLP thiên Fake nên Accuracy thấp dù AUC cao nhất. Đây là giới hạn chung của các mô hình hiện tại và gợi ý hướng tiếp theo là calibration hoặc threshold adaptation theo thời gian.

5.6. Đánh giá rò rỉ thông tin từ FastRP

FastRP là đặc trưng cấu trúc mạnh, nhưng trong thiết lập OOD quy nạp nó có thể gây rò rỉ nếu được tính trên toàn bộ đồ thị trước khi tách train-test. Khi giữ FastRP trong đầu vào GNN, nhánh chẩn đoán nội bộ và CausalHeteroGNN đạt OOD lần lượt 93.6±1.7% và 94.3±1.9%, với AUC xấp xỉ 0.980 và F1-drop gần như bằng 0. Kết quả này trông lý tưởng nhưng thực chất phản ánh rò rỉ cấu trúc.

Khi loại bỏ FastRP để đảm bảo thiết lập quy nạp thực sự, hiệu năng OOD giảm xuống 61.0±2.1% với nhánh chẩn đoán nội bộ và 61.1±1.6% với CausalHeteroGNN trong thí nghiệm này; AUC còn khoảng 0.630; F1-drop tăng lên 25.1% và 25.2%. Khoảng cách hơn 33 điểm phần trăm cho thấy phần lớn hiệu năng cao bất thường khi dùng FastRP đến từ thông tin cấu trúc toàn cục đã mã hóa gián tiếp nhãn hoặc cộng đồng OOD.

Hình 5. Tác động rò rỉ đặc trưng cấu trúc FastRP đến kết quả OOD.

Kết luận thực nghiệm là FastRP không nên được đưa vào input mặc định nếu mục tiêu là đánh giá quy nạp công bằng. Nó có thể được dùng trong dashboard BI hoặc phân tích cấu trúc, nhưng cần tách rõ khỏi pipeline học máy chính.

5.7. Phân tích can thiệp cấu trúc qua Label-Flip Rate

Nghiên cứu thực hiện các phép can thiệp do(.) trên đồ thị và đo tỷ lệ thay đổi dự đoán nhãn của mô hình (Label-Flip Rate - LFR).

Bảng 7. Tỷ lệ lật nhãn dưới các phép can thiệp cấu trúc.

| Phép can thiệp | LFR Baseline | LFR Causal | Diễn giải |
|---|---:|---:|---|
| do(C1 = swap) | 30.1% | ~0.0% | Hoán đổi thông tin cộng đồng |
| do(I = empty) | 4.0% | 8.0% | Loại bỏ đặc trưng hình ảnh |
| do(D = credible) | 6.2% | 15.5% | Thay đổi độ tin cậy nguồn tin |

LFR gần 0% dưới do(C1=swap) là một sanity check đúng theo thiết kế: nhánh causal đã cắt các cạnh Subreddit nên dự đoán không thay đổi khi cộng đồng bị hoán đổi. Ngược lại, việc LFR tăng với do(I=empty) và do(D=credible) cho thấy khi giảm phụ thuộc vào cộng đồng, mô hình nhạy hơn với nội dung hình ảnh và độ tin cậy nguồn tin.

6. Discussion - Thảo luận

Các kết quả cho thấy ba điểm chính. Thứ nhất, Held-Out Subreddit một fold là chưa đủ để đánh giá robust OOD: các mô hình đạt hiệu năng gần nhau, dễ dẫn đến kết luận rằng can thiệp nhân quả không cần thiết. Thứ hai, Confounding-Shift cho thấy khi shortcut cộng đồng bị đảo có kiểm soát, Baseline GNN và các regularizer mềm như IRM/EERM không đủ ổn định, trong khi can thiệp cấu trúc đạt hiệu quả cao hơn rõ rệt. Thứ ba, LOCO chứng minh shortcut cộng đồng cũng gây hại trong dữ liệu tự nhiên, vì Baseline GNN suy giảm trên mọi fold trong khi CausalHeteroGNN giữ hiệu năng gần với content-only ceiling.

So với hiện trạng nghiên cứu, đóng góp chính của nghiên cứu này không nằm ở việc tuyên bố state-of-the-art chung trên Fakeddit, mà ở việc kiểm toán và kiểm soát một loại shortcut cụ thể trong HGNN đa phương thức. Cách tiếp cận này bổ sung cho các hướng hiện có như invariant risk minimization, graph OOD learning, causal subgraph extraction và multimodal benchmark mới. Điểm khác biệt là can thiệp được thực hiện trực tiếp trên loại quan hệ gây nhiễu đã được xác định bằng phân tích dữ liệu và SCM, sau đó được kiểm tra bằng stress-test, counterfactual LFR, worst-group và split tự nhiên.

Nghiên cứu cũng chỉ ra các giới hạn cần trình bày minh bạch. Temporal split cho thấy chưa mô hình nào xử lý tốt prior shift theo thời gian; nhóm satire như r/theonion vẫn là nhóm khó trong Standard OOD; và quy mô thí nghiệm 5,898 bài nhỏ hơn nhiều so với toàn bộ Fakeddit. Các giới hạn này không phủ định kết quả chính, nhưng giúp định vị phương pháp như một bước kiểm soát shortcut cộng đồng thay vì lời giải hoàn chỉnh cho mọi dạng dịch chuyển phân phối.

7. Conclusion - Kết luận

Nghiên cứu đã đề xuất CausalHeteroGNN, một mô hình Heterogeneous GraphSAGE theo hướng nhân quả cho phát hiện tin giả đa phương thức. Bằng cách xác định Subreddit là biến gây nhiễu và loại bỏ các đường truyền thông tin liên quan đến Subreddit trong nhánh causal, mô hình hạn chế việc học shortcut cộng đồng.

Trên Confounding-Shift Benchmark, CausalHeteroGNN đạt 79.9±4.2% OOD Accuracy, AUC 0.922, F1-drop 5.4% và worst-group accuracy 37.8±11.3%, vượt Baseline GNN (52.1±7.8%, worst-group 23.5±7.9%), IRM (54.1±4.7%, worst-group 24.4±7.1%), MLP content-only (59.5±1.7%, worst-group 15.6±3.1%) và EERM (59.6±1.1%, worst-group 28.6±1.3%). Trên LOCO tự nhiên, CausalHeteroGNN đạt 69.4±5.4%, cao hơn Baseline GNN 56.5±1.3% và IRM 65.8±2.9%; EERM đạt 70.0±6.0% (3 seeds) tương đương MLP 71.6±7.9% nhưng dao động lớn hơn. Những kết quả này cho thấy mô hình đồ thị thuần dễ bị shortcut cộng đồng làm suy giảm, trong khi can thiệp cấu trúc giúp giữ lại lợi ích của đồ thị một cách bền vững hơn.

Nghiên cứu cũng lượng hóa rủi ro rò rỉ FastRP: khi đặc trưng này được tính trên toàn bộ đồ thị trước khi tách train-test, OOD Accuracy có thể tăng ảo lên hơn 93-94%; khi loại bỏ FastRP, hiệu năng trở về khoảng 61%, phản ánh đúng độ khó của thiết lập quy nạp. Trong tương lai, nghiên cứu có thể mở rộng lên toàn bộ Fakeddit, cải thiện nhận diện satire, phát triển calibration cho prior shift theo thời gian, và tự động hóa việc phát hiện confounder bằng các tiêu chí chọn quan hệ không nhìn OOD.

Tài liệu tham khảo (References)

[1] K. Nakamura, S. Levy, and W. Y. Wang, "r/Fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection," in Proc. 12th Lang. Resour. Eval. Conf. (LREC), Marseille, France, 2020, pp. 6149-6157.

[2] J. Pearl, Causality: Models, Reasoning, and Inference, 2nd ed. Cambridge, U.K.: Cambridge Univ. Press, 2009.

[3] W. L. Hamilton, R. Ying, and J. Leskovec, "Inductive representation learning on large graphs," in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 1024-1034.

[4] Z. Hu, Y. Dong, K. Wang, and Y. Sun, "Heterogeneous graph transformer," in Proc. Web Conf. (WWW), 2020, pp. 2704-2710.

[5] Y. Ganin et al., "Domain-adversarial training of neural networks," J. Mach. Learn. Res., vol. 17, no. 59, pp. 1-35, 2016.

[6] B. Schölkopf, "Causality for machine learning," in Probabilistic and Causal Inference: The Works of Judea Pearl. New York, NY, USA: ACM, 2022, pp. 765-804.

[7] R. Ying, D. Bourgeois, J. You, M. Zitnik, and J. Leskovec, "GNNExplainer: Generating explanations for graph neural networks," in Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 9240-9251.

[8] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz, "Invariant risk minimization," 2019, arXiv:1907.02893.

[9] J. Peters, D. Janzing, and B. Schölkopf, Elements of Causal Inference: Foundations and Learning Algorithms. Cambridge, MA, USA: MIT Press, 2017.

[10] I. Gulrajani and D. Lopez-Paz, "In search of lost domain generalization," in Proc. Int. Conf. Learn. Representations (ICLR), 2021.

[11] T. Bian, X. Xiao, T. Xu, P. Zhao, W. Huang, Y. Rong, and J. Huang, "Rumor detection on social media with bi-directional graph convolutional networks," in Proc. AAAI Conf. Artif. Intell., vol. 34, no. 01, 2020, pp. 549-556.

[12] X. Zhou, J. Wu, and R. Zafarani, "SAFE: Similarity-aware multi-modal fake news detection," in Proc. Pacific-Asia Conf. Knowl. Discovery Data Mining (PAKDD), 2020, pp. 354-367.

[13] Q. Wu, H. Zhang, J. Yan, and D. Wipf, "Handling distribution shifts on graphs: An invariance perspective," in Proc. Int. Conf. Learn. Representations (ICLR), 2022.

[14] Z. Liu, C. Xiong, M. Sun, and Z. Liu, "Fine-grained fact verification with kernel graph attention network," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 7342-7351.

[15] I. Robinson, J. Webber, and E. Eifrem, Graph Databases: New Opportunities for Connected Data, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2015.

[16] J. Lv, Y. Gao, L. Li, L. Shi, and S. Li, "Multi-modal fake news detection: A comprehensive survey on deep learning technology, advances, and challenges," Journal of King Saud University - Computer and Information Sciences, vol. 37, article 306, 2025.

[17] X. Liu et al., "MMFakeBench: A mixed-source multimodal misinformation detection benchmark for LVLMs," in Proc. Int. Conf. Learn. Representations (ICLR), 2025.

[18] S. Gong, R. O. Sinnott, J. Qi, and C. Paris, "Invariant subgraphs for cross-domain fake news detection via causal disentanglement," Proc. Int. AAAI Conf. Web and Social Media (ICWSM), vol. 20, no. 1, pp. 910-922, 2026.
