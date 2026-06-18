# 05 — Ví dụ minh họa: Một bài post đi qua toàn hệ thống

> **Câu hỏi thầy:** "Nên lấy một ví dụ thực tế về mặt dữ liệu khi đi vào hệ thống:
> đồ thị hình thù ra sao, từng bước biến đổi thế nào để ra được kết quả cuối cùng."

---

## Ví dụ: Bài post từ r/theonion

Giả sử ta có bài post **thật** trong tập test với thông tin:

```
post_id:       "abc123"
title:         "Scientists Discover New Species Of Bird That Hates Everything"
subreddit:     "theonion"
author:        "satirist_writer"
domain:        "theonion.com"
image_url:     "https://theonion.com/img/abc123.jpg"
2_way_label:   0   → FAKE  (nhãn thật sự)
6_way_label:   1   → Satire
split:         "test"
is_ood:        True  (subreddit theonion KHÔNG có trong train)
```

---

## BƯỚC 1: Chuẩn bị dữ liệu (01_prepare_data.py)

### 1a. Tải ảnh
```python
download_single_image("abc123", "https://theonion.com/img/abc123.jpg")
# Lưu: data/images/abc123.jpg
```

### 1b. Trích xuất CLIP embedding cho ảnh
```python
img = Image.open("data/images/abc123.jpg").convert("RGB")
clip_inputs = clip_processor(images=[img], return_tensors="pt")
img_feat = clip_model.vision_model(pixel_values=...)
clip_emb = normalize(visual_projection(img_feat))
# Kết quả: vector 512 chiều
# Ví dụ: [-0.02, 0.15, -0.08, 0.31, ..., 0.07]  (512 số)
```

### 1c. Encode tiêu đề bằng SentenceTransformer
```python
text_emb = model.encode("Scientists Discover New Species Of Bird That Hates Everything")
# Kết quả: vector 768 chiều, L2-normalized
# Ví dụ: [0.03, -0.12, 0.08, 0.25, ..., -0.01]  (768 số)
```

### 1d. Ghi vào CSV
```
posts.csv:
  post_id=abc123, title="Scientists...", score=1847, upvote_ratio=0.91,
  num_comments=234, label_2way=0, split=test, is_ood=True

images.csv:
  img_id=img_abc123, post_id=abc123, has_image=True

posted_in.csv:
  post_id=abc123, sub_id=sub_theonion   ← cạnh Post→Subreddit

links_to.csv:
  post_id=abc123, domain_id=domain_theonion.com
```

---

## BƯỚC 2: Nạp vào Neo4j và chạy GDS (02_neo4j_import.py)

```cypher
MERGE (p:Post {post_id: "abc123"})
MERGE (i:Image {img_id: "img_abc123"})
MERGE (p)-[:POSTED_IN]->(s:Subreddit {sub_id: "sub_theonion"})
MERGE (p)-[:LINKS_TO]->(d:Domain {domain_id: "domain_theonion.com"})
MERGE (p)-[:HAS_IMAGE]->(i)
```

Sau khi import, GDS tính PageRank cho `domain_theonion.com` và Louvain community cho Post.

---

## BƯỚC 3: Xây dựng HeteroData cho GNN (05_train_gnn.py)

### 3a. Feature vector của Post "abc123"
```python
p_scalar = normalize([score=1847, upvote_ratio=0.91, num_comments=234])
         → [0.73, 0.85, 0.46]  (sau min-max norm)

Post.x["abc123"] = concat([text_768d, p_scalar_3d])
                 = tensor của shape (771,)
```

### 3b. Feature của các node láng giềng
```
User "satirist_writer":
  x = [post_count=23, avg_score=850.0, avg_upvote_ratio=0.88, fake_rate=0.5]
  → (chỉ từ train, nhưng satirist_writer KHÔNG có trong train → dùng default 0.5)

Subreddit "theonion":
  x = [post_count=1, fake_ratio_real=0.5, avg_score=median_train]
  → (theonion không có trong train → dùng default neutral 0.5)

Domain "theonion.com":
  x = [fake_ratio_real=0.5, avg_upvote_ratio=median, post_count=1]
  → (theonion.com không có trong train → dùng default)

Image "img_abc123":
  x = CLIP embedding (512,)
```

### 3c. Cạnh (edge_index) liên quan đến "abc123"
```
Post→User:      edge_index[:,k] = [post_idx_abc123, user_idx_satirist]
Post→Subreddit: edge_index[:,m] = [post_idx_abc123, sub_idx_theonion]
Post→Domain:    edge_index[:,n] = [post_idx_abc123, dom_idx_theonion_com]
Post→Image:     edge_index[:,p] = [post_idx_abc123, img_idx_abc123]
```

---

## BƯỚC 4: Lan truyền thông điệp (forward pass)

### 4a. Projection ban đầu

```
h₀["abc123"] = ReLU(Linear(771→96)(Post.x["abc123"]))
             = vector (96,) — không gian ẩn chung
```

Tương tự cho tất cả User, Subreddit, Domain, Image nodes.

### 4b. GraphSAGE lớp 1 — Nhánh G_causal (cạnh Subreddit đã bị cắt)

```
Neighbors hợp lệ của abc123 trong G_causal:
  - user "satirist_writer"    (qua POSTED_BY)
  - domain "theonion.com"     (qua LINKS_TO)
  - image "img_abc123"        (qua HAS_IMAGE)
  KHÔNG có: subreddit "theonion" (đã bị cắt)

Message từ User:   m_user   = SAGEConv(h₀[satirist_writer])
Message từ Domain: m_domain = SAGEConv(h₀[theonion.com])
Message từ Image:  m_image  = SAGEConv(h₀[img_abc123])

h₁["abc123"] = Aggregate([m_user, m_domain, m_image])
             = SAGEConv sums messages → vector (96,)
```

> Vì subreddit "theonion" KHÔNG có trong train, feature của nó là neutral (0.5).
> Nhưng dù sao cạnh này cũng đã bị cắt ở G_causal → Subreddit không gửi message.

### 4c. GraphSAGE lớp 2 (2-hop aggregation)

```
h₂["abc123"] = SAGEConv(h₁["abc123"], 2-hop neighbors causal)
             = vector (96,)  — đã hấp thụ thông tin 2-hop
```

### 4d. causal_head và classifier

```
h_causal["abc123"] = MLP(h₂["abc123"])
                   = [0.34, -0.12, 0.78, ..., -0.05]  (96 chiều)

logits = causal_clf_2way(h_causal["abc123"])
       = [2.31, -0.87]  (2 chiều: [score_Fake, score_Real])

Softmax → [p(Fake)=0.96, p(Real)=0.04]
argmax  → 0 → ŷ = FAKE ✓  (khớp với nhãn thật)
```

---

## BƯỚC 5: Giải thích nhân quả (causal path attribution)

Dùng gradient để tính mức độ đóng góp của từng nguồn:

```python
prob_fake = Softmax(logits)[0]  # = 0.96
prob_fake.backward()             # tính gradient

grad_subreddit = data["Subreddit"].x.grad[sub_idx_theonion].abs().sum()  # = 0.0 (bị cắt)
grad_domain    = data["Domain"].x.grad[dom_idx_theonion_com].abs().sum() # = 2.3
grad_image     = data["Image"].x.grad[img_idx_abc123].abs().sum()        # = 1.5

total = 0 + 2.3 + 1.5 = 3.8
attribution_domain = 2.3 / 3.8 = 60.5%
attribution_image  = 1.5 / 3.8 = 39.5%
```

**Kết quả giải thích:**
```
"Flagged chủ yếu do nguồn domain. Domain 'theonion.com' có fake_ratio=0.50
và chiếm 60.5% influence. Image chiếm 39.5%."
```

---

## BƯỚC 6: Counterfactual — "What if?"

**CF1: do(image=None)** — Xóa cạnh Post→Image, chạy lại

```
Logits mới: [1.87, -0.45] → p(Fake)=0.88
Kết luận: vẫn Fake → Image không phải yếu tố quyết định duy nhất
```

**CF2: do(domain=credible)** — Thay domain thành reuters.com (domain uy tín nhất)

```
Logits mới: [0.12, 1.34] → p(Fake)=0.18 → chuyển sang Real!
Label Flip Rate (LFR_domain) tăng lên → xác nhận Domain có tác động nhân quả lớn
```

**CF3: do(subreddit=neutral)** — Thay subreddit thành r/news (subreddit trung tính)

```
[Ở nhánh causal] Subreddit đã bị cắt → thay subreddit không làm thay đổi logits nhiều
[Ở nhánh baseline] Có thể thay đổi → thể hiện baseline "học tắt" qua subreddit
```

---

## Tóm tắt trực quan

```
[Bài post: "Scientists Discover..."]
                │
                ▼
  ┌─────────────────────────────┐
  │  Feature của Post:          │
  │  text_emb: (768,)           │
  │  score=1847, ratio=0.91     │
  └─────────────┬───────────────┘
                │
                ▼ GraphSAGE (G_causal — không Subreddit)
  ┌─────────────────────────────┐
  │  Messages từ:               │
  │  ✓ User satirist_writer     │
  │  ✓ Domain theonion.com      │
  │  ✓ Image (CLIP 512-d)       │
  │  ✗ Subreddit theonion (cắt) │
  └─────────────┬───────────────┘
                │
                ▼ h_causal (96-d)
  ┌─────────────────────────────┐
  │  Classifier (MLP):          │
  │  logits = [2.31, -0.87]     │
  │  softmax = [96%, 4%]        │
  │  argmax = 0                 │
  └─────────────┬───────────────┘
                │
                ▼
          ŷ = FAKE  ✓
```
