# Huong dan giai thich Hinh 2, bo phan loai va doi chieu code

Tai lieu nay duoc viet de tra loi truc tiep cac cau hoi gop y: kien truc Hinh 2, ban chat bo phan loai, vi du minh hoa du lieu di qua he thong, va doi chieu tung buoc voi ma nguon da chay.

## 1. Kien truc mo hinh trong Hinh 2

### 1.1. Du lieu dau vao: tu Fakeddit goc den tap huan luyen

Nguon du lieu ban dau la cac file TSV cua Fakeddit, chu yeu gom `multimodal_train.tsv` va `multimodal_validate.tsv`. Script [pipeline/01_prepare_data.py](pipeline/01_prepare_data.py) thuc hien cac buoc:

1. Doc file TSV goc bang `pandas.read_csv(..., sep="\t")`.
2. Loc cac bai co anh that: `hasImage == True`, `image_url` ton tai va bat dau bang `http`.
3. Tach hai subreddit OOD (`neutralnews`, `theonion`) ra khoi train/validation. Hai subreddit nay chi xuat hien trong test OOD, giup kiem tra kha nang tong quat hoa.
4. Lay mau can bang theo nhan 2-way: train gom fake/real can bang, validation can bang, test gom seen-test va OOD-test.
5. Tai anh ve `data/images/`.
6. Tao embedding van ban cho tieu de bang `SentenceTransformer` (`all-mpnet-base-v2`).
7. Tao embedding anh bang CLIP (`openai/clip-vit-base-patch32`).
8. Tinh cac thong ke User/Subreddit/Domain chi tu split train de tranh ro ri nhan test.
9. Xuat CSV node/edge va cac file `.npy` de dung cho Neo4j va PyTorch Geometric.

Noi cach khac, tap huan luyen khong duoc dua truc tiep vao mo hinh duoi dang bang phang. Moi dong bai dang duoc bien thanh mot nut `Post`, roi noi voi cac nut ngu canh nhu `User`, `Subreddit`, `Domain`, `Image`.

### 1.2. Xay dung do thi di the co 5 loai nut

Ban chat cua buoc xay dung do thi la bien moi mau du lieu thanh mot property graph:

| Thanh phan | Du lieu nguon | File CSV | Vai tro |
|---|---|---|---|
| Post | id, title, score, upvote_ratio, label | `posts.csv` | Doi tuong can phan loai Fake/Real |
| User | author va thong ke train-only | `users.csv` | Ngu canh nguoi dang |
| Subreddit | subreddit va fake_ratio train-only | `subreddits.csv` | Cong dong, confounder chinh |
| Domain | domain va fake_ratio train-only | `domains.csv` | Nguon tin/lien ket |
| Image | image_url, post_id | `images.csv` | Anh di kem bai dang |

Nam loai canh duoc tao:

| Canh | Y nghia | File |
|---|---|---|
| `POSTED_BY` | Post do User nao dang | `posted_by.csv` |
| `POSTED_IN` | Post nam trong Subreddit nao | `posted_in.csv` |
| `LINKS_TO` | Post lien ket toi Domain nao | `links_to.csv` |
| `HAS_IMAGE` | Post co Image nao | `has_image.csv` |
| `MEMBER_OF` | User hoat dong trong Subreddit nao | `member_of.csv` |

Cong cu thuc te:

- `pandas` de tao bang node/edge CSV.
- `Neo4j` de import property graph va kiem tra luoc do.
- `Neo4j GDS` de tinh PageRank, Louvain, Betweenness, FastRP phuc vu phan tich/dashboard.
- `NetworkX` la fallback neu Neo4j/GDS khong chay.
- `torch_geometric.data.HeteroData` de chuyen do thi CSV thanh tensor graph dua vao GNN.

Trong code, `02_neo4j_import.py` import vao Neo4j. Sau do `05_train_gnn.py::build_heterodata()` doc lai CSV, tao `HeteroData`, lap index map cho tung loai nut, tao `edge_index` cho tung quan he, roi goi `T.ToUndirected()` de them canh nguoc. Day la cau truc thuc su ma GNN nhan vao.

### 1.3. Box `HeteroData Input`

Box nay la dau vao cua mo hinh GNN, khong phai du lieu TSV tho. `HeteroData` gom:

- `data["Post"].x`: dac trung Post, gom text embedding + cac scalar nhu score/upvote/num_comments, tuy cau hinh co the them `clip_cons`.
- `data["Image"].x`: embedding anh CLIP 512 chieu.
- `data["User"].x`, `data["Subreddit"].x`, `data["Domain"].x`: cac thong ke duoc tinh tu train.
- `data["Post"].y`: nhan 2-way Fake/Real.
- `data["Post"].y_6way`: nhan 6-way cua Fakeddit.
- `data[edge_type].edge_index`: danh sach canh dang tensor kich thuoc `[2, num_edges]`.

### 1.4. Box `G - Do thi di the day du`

`G` la do thi giu nguyen moi quan he: Post-User, Post-Subreddit, Post-Domain, Post-Image va User-Subreddit. Nhanh baseline encode tren `G` de cho thay mo hinh co the khai thac day du ngu canh, bao gom ca shortcut Subreddit.

Trong code, nhanh baseline la:

```python
h_full = self.encode(x_dict, edge_index_dict)
h_post = h_full["Post"]
```

### 1.5. Box `G_causal - Do thi can thiep`

`G_causal` la phien ban do thi da can thiep. Can thiep chinh la xoa moi canh co nut nguon hoac nut dich la `Subreddit`:

```text
E_causal = { e = (u, v) in E : type(u) != Subreddit va type(v) != Subreddit }
```

Y nghia nhan qua: Subreddit la bien gay nhieu cau truc, vi no vua anh huong phong cach/noi dung bai viet, vua tuong quan rat manh voi nhan Fake/Real trong train. Neu cho GNN truyen thong diep qua Subreddit, mo hinh co the hoc "bai o subreddit A thi fake" thay vi hoc noi dung. Cat canh Subreddit la mot phep can thiep cau truc de chan duong shortcut.

Trong code:

```python
def _cut_confounder_edges(edge_index_dict):
    return {k: v for k, v in edge_index_dict.items()
            if k[0] != "Subreddit" and k[2] != "Subreddit"}
```

### 1.6. Vi sao can tach hai phien ban do thi?

Can tach `G` va `G_causal` vi chung tra loi hai cau hoi khac nhau:

- `G`: Neu giu day du ngu canh, GNN hoc duoc gi? No co the hoc ca tin hieu dung lan shortcut.
- `G_causal`: Neu chan Subreddit shortcut, GNN con dua vao noi dung, anh, domain, user duoc khong?

Hai do thi khong phai hai dataset khac nhau. Chung co cung tap nut, cung dac trung nut, cung nhan; chi khac tap canh duoc phep truyen thong diep trong encoder. Nho vay, so sanh giua hai nhanh la so sanh tac dong cua can thiep cau truc, khong phai do thay doi du lieu.

### 1.7. "Hoa nhap" hoac doi sanh hai do thi la gi?

Trong code khong co buoc hop nhat hai do thi thanh mot do thi moi. "Hoa nhap/doi sanh" nen giai thich la:

1. Hai nhanh dung chung `x_dict`, tuc cung node features.
2. Hai nhanh dung chung encoder GraphSAGE weights, tuc cung cach hoc thong diep.
3. Hai nhanh tra ve embedding cho cung thu tu nut `Post`.
4. Embedding cua cung mot Post duoc dua vao cac head khac nhau: `h_spurious` va `h_causal`.
5. Loss tong ket hop cac dau ra de cap nhat mot bo tham so chung.

Vay "doi sanh" la doi sanh theo chi so nut Post: Post thu i trong nhanh full graph va Post thu i trong nhanh causal cung la mot bai dang, chi khac ngu canh message passing.

### 1.8. Box `Heterogeneous GraphSAGE Encoder`

Encoder la hai lop HeteroGraphSAGE:

- Moi loai nut duoc chieu ve hidden dimension `d = 96` bang `nn.Linear`.
- Voi moi loai quan he, PyTorch Geometric tao mot `SAGEConv`.
- `HeteroConv(..., aggr="sum")` gom thong diep tu cac quan he khac nhau.
- Encoder duoc goi hai lan: mot lan tren `G`, mot lan tren `G_causal`.

Cong thuc rut gon:

```text
h_v^(l+1) = sigma(W_self h_v^(l) + AGG_r({W_r h_u^(l) : u in N_r(v)}))
```

Trong code, `SAGEConv` va `HeteroConv` la thu vien PyTorch Geometric; viec cat canh, chia hai nhanh, va loss nhan qua la logic tu xay dung.

### 1.9. Box `h_spurious` va `h_causal`

- `h_spurious` duoc sinh tu nhanh do thi day du, nen co kha nang chua tin hieu Subreddit/shortcut.
- `h_causal` duoc sinh tu nhanh da cat Subreddit, nen duoc ky vong chua tin hieu ben vung hon.

Hai bieu dien bi rang buoc bang `L_ortho` de giam viec hoc trung nhau:

```text
L_ortho = mean(|cos(h_causal, h_spurious)|)
```

### 1.10. Box `GRL` va bo phan loai Subreddit

GRL la Gradient Reversal Layer. Khi forward, no giu nguyen vector. Khi backward, no dao dau gradient. Trong mo hinh:

- Bo phan loai Subreddit co gang doan Post thuoc subreddit nao tu `h_causal`.
- Do gradient bi dao, encoder lai hoc lam cho `h_causal` kho doan Subreddit hon.

Day la co che adversarial de loai bot tin hieu cong dong con sot lai sau khi cat canh.

## 2. Ban chat cua "bo phan loai" (Classifier)

### 2.1. Classifier co hinh dang gi?

Trong code, classifier khong phai la mot do thi. No la mot mang neural network dang MLP (Multi-Layer Perceptron). Ham tao classifier:

```python
def mlp(in_dim, out_dim, dropout_p=dropout):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim),
    )
```

Voi 2-way Fake/Real, `out_dim = 2`. Dau ra la hai logit:

```text
z = [z_fake, z_real]
```

Xac suat duoc tinh bang softmax:

```text
P(y=k|v) = exp(z_k) / sum_j exp(z_j)
```

Ket luan cuoi cung:

```text
y_hat = argmax_k P(y=k|v)
```

Neu index 0 la Fake va index 1 la Real thi:

- `z_fake > z_real` -> du doan Fake.
- `z_real > z_fake` -> du doan Real.

### 2.2. Co nhung classifier nao trong model?

Trong `CausalHeteroGNN` co bon classifier chinh va mot classifier phu:

| Ten trong code | Dau vao | Dau ra | Muc dich |
|---|---|---|---|
| `baseline_clf_2way` | `h_post` tu G day du | 2 logit | Chan doan Fake/Real tren full graph |
| `causal_clf_2way` | `h_causal` | 2 logit | Dau ra chinh Fake/Real |
| `baseline_clf_6way` | `h_post` | 6 logit | Nhiem vu phu 6-way tren full graph |
| `causal_clf_6way` | `h_causal` | 6 logit | Nhiem vu phu 6-way cho nhanh causal |
| `confounder_clf` | `h_spurious` hoac `GRL(h_causal)` | so subreddit | Doan confounder/Subreddit |

Khi trinh bay voi thầy, nen nhan manh: bo phan loai Fake/Real la MLP nhan embedding Post sau GNN, khong phai Neo4j, khong phai mot do thi moi.

### 2.3. Classifier dung de kiem thu test set nhu the nao?

Quy trinh inference:

1. Nap checkpoint da train tu `models/causal_gnn*.pt`.
2. Tao lai `HeteroData` voi dung thu tu node/features.
3. Chay forward:

```python
test_out_base_2, test_out_causal_2, ... = model(data.x_dict, data.edge_index_dict)
```

4. Lay cac hang ung voi `test_mask`.
5. Tinh softmax/logit va lay `argmax`.
6. So sanh voi nhan that `data["Post"].y[test_mask]`.
7. Tinh accuracy, macro-F1, precision, recall, AUC, confusion matrix.

Trong bai viet, ket qua chinh nen lay tu `causal_clf_2way`, vi day la dau ra cua nhanh nhan qua. Nhanh baseline chi nen xem la diagnostic.

### 2.4. Co so dua ra Fake hay Real

Vi du mot Post sau encoder co vector `h_causal`. MLP sinh:

```text
z = [-0.8, 1.4]
softmax(z) = [0.10, 0.90]
```

Neu quy uoc `0 = Fake`, `1 = Real`, model ket luan Real voi xac suat 0.90.

Nguoc lai:

```text
z = [2.1, -0.4]
softmax(z) = [0.92, 0.08]
```

Model ket luan Fake voi xac suat 0.92.

## 3. Vi du minh hoa du lieu di qua he thong

Gia su co mot bai Reddit:

```text
id: p123
title: "Breaking: Celebrity endorses miracle cure"
author: alice
subreddit: theonion
domain: example-news.com
image_url: https://.../p123.jpg
label_2way: Fake
```

### 3.1. Sau buoc chuan bi du lieu

Pipeline tao cac nut:

```text
(Post {post_id: p123, title: ..., label_2way: 0})
(User {user_id: user_alice, post_count: ..., fake_rate: ...})
(Subreddit {sub_id: sub_theonion, fake_ratio_real: ...})
(Domain {domain_id: domain_example-news.com, fake_ratio_real: ...})
(Image {img_id: img_p123, image_url: ...})
```

Va cac canh:

```text
(p123)-[:POSTED_BY]->(user_alice)
(p123)-[:POSTED_IN]->(sub_theonion)
(p123)-[:LINKS_TO]->(domain_example-news.com)
(p123)-[:HAS_IMAGE]->(img_p123)
(user_alice)-[:MEMBER_OF]->(sub_theonion)
```

### 3.2. Khi vao GNN

`Post p123` co vector dac trung gom:

- embedding tieu de tu SentenceTransformer;
- `score`, `upvote_ratio`, `num_comments`;
- neu bat cau hinh, co them `clip_cons`;
- anh `img_p123` co vector CLIP rieng trong node Image.

Trong nhanh `G`, Post p123 nhan thong diep tu User, Subreddit, Domain, Image. Neu Subreddit co lien he nhan rat manh, model co the hoc shortcut.

Trong nhanh `G_causal`, cac canh den Subreddit bi loai. Post p123 khong the lay thong diep truc tiep tu `sub_theonion`. Model phai dua nhieu hon vao title, image, domain va cac dac trung con lai.

### 3.3. Ra ket qua

Sau hai lop HeteroGraphSAGE:

```text
h_causal(p123) -> causal_clf_2way -> [logit_fake, logit_real]
```

Neu softmax cho:

```text
P(Fake) = 0.87, P(Real) = 0.13
```

thi model ket luan `Fake`.

## 4. Doi chieu Hinh 2 voi code thuc te

| Box Hinh 2 | Code doi chieu | Giai thich |
|---|---|---|
| HeteroData Input | `05_train_gnn.py::build_heterodata()` | Doc CSV/NPY va tao `HeteroData` |
| Post/User/Subreddit/Domain/Image | `01_prepare_data.py::create_neo4j_csvs()` | Tao 5 loai node tu Fakeddit |
| G day du | `forward(...): self.encode(x_dict, edge_index_dict)` | Message passing tren moi canh |
| G_causal | `_cut_confounder_edges()` | Xoa canh co nguon/dich la Subreddit |
| Heterogeneous GraphSAGE Encoder | `self.conv1`, `self.conv2` | `HeteroConv` + `SAGEConv` cua PyTorch Geometric |
| h_spurious | `self.spurious_head(h_post)` | Bieu dien phu tu full graph |
| h_causal | `self.causal_head(h_post_caus)` | Bieu dien chinh tu causal graph |
| GRL | `GradientReversal`, `GRL` | Dao gradient khi hoc loai thong tin Subreddit |
| Subreddit classifier | `self.confounder_clf` | Doan subreddit tu bieu dien, dung cho adversarial loss |
| Fake/Real classifier | `self.causal_clf_2way` | MLP dua ra 2 logit Fake/Real |
| L_ortho | `F.cosine_similarity(h_c, h_s)` | Ep hai bieu dien khac nhau |
| Test/evaluation | `06_evaluate.py` | Nap checkpoint, forward, tinh metrics |

## 5. Phan nao dung thu vien, phan nao tu xay dung?

| Phan | Thu vien | Tu xay dung trong de tai |
|---|---|---|
| Doc/loc/lay mau du lieu | `pandas`, `numpy` | Quy tac split, OOD subreddit, train-only features |
| Tai anh | `requests`, `ThreadPoolExecutor` | Dieu kien loc anh hop le |
| Text embedding | `sentence-transformers` | Cach gan embedding vao Post |
| Image embedding | `transformers` CLIP, `PIL`, `torch` | Fail-hard neu embedding loi |
| Neo4j import | `neo4j` driver | Luoc do 5 node/5 edge, Cypher batch |
| GDS | Neo4j GDS | Chon PageRank/Louvain/Betweenness/FastRP |
| Fallback graph | `networkx` | FastRP-style local va CSV enriched |
| Hetero graph tensor | PyTorch Geometric `HeteroData` | Mapping CSV -> node index -> edge_index |
| Encoder | PyG `HeteroConv`, `SAGEConv` | Hai nhanh full/causal dung chung encoder |
| Classifier | PyTorch `nn.Linear`, `nn.Sequential` | Thiet ke multi-head 2-way/6-way/confounder |
| Causal cut | Khong co san | `_cut_confounder_edges()` |
| GRL | PyTorch autograd | `GradientReversal` tu dinh nghia |
| Counterfactual | PyTorch tensor ops | Sua edge_index de mo phong do(image/domain/subreddit) |

## 6. Doan dien giai co the dua vao bai viet

Kien truc CausalHeteroGNN nhan dau vao la mot `HeteroData` gom 5 loai nut: Post, User, Subreddit, Domain va Image. Moi bai dang trong Fakeddit duoc bieu dien thanh mot nut Post va duoc noi voi nguoi dang, cong dong dang tai, mien nguon va anh di kem thong qua 5 loai quan he. Cac dac trung van ban va hinh anh duoc tao bang SentenceTransformer va CLIP, trong khi cac dac trung lich su cua User/Subreddit/Domain duoc tinh rieng tu tap train de tranh ro ri thong tin test.

Tu cung mot do thi dau vao, mo hinh tao hai phien ban truyen thong diep. Phien ban thu nhat la do thi day du G, giu moi quan he va duoc dung cho nhanh baseline/spurious. Phien ban thu hai la do thi can thiep G_causal, trong do moi canh lien quan den Subreddit bi loai bo. Viec loai bo nay nham chan duong shortcut qua cong dong, vi Subreddit trong Fakeddit co tuong quan rat manh voi nhan Fake/Real. Hai nhanh dung chung dac trung nut va encoder HeteroGraphSAGE; chung chi khac tap canh duoc phep truyen thong diep. Do do, embedding cua cung mot Post o hai nhanh co the duoc doi sanh truc tiep.

Bo phan loai Fake/Real la mot MLP nam sau embedding Post, khong phai mot do thi rieng. MLP nhan vector `h_causal` va sinh hai logit tuong ung Fake va Real. Xac suat duoc tinh bang softmax, va nhan du doan la lop co xac suat lon nhat. Trong qua trinh danh gia, checkpoint da huan luyen duoc nap lai, model forward tren test set, lay cac Post co `test_mask`, roi so sanh `argmax(softmax(logits))` voi nhan that de tinh accuracy, macro-F1 va AUC.

