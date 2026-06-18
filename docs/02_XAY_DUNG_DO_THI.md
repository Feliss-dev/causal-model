# 02 — Xây dựng đồ thị dị thể (Heterogeneous Information Network)

> **Câu hỏi thầy:** "Bản chất việc xây dựng đồ thị là gì? Dùng công cụ hay thuật toán gì?
> Cách chuyển đổi dữ liệu thành đồ thị dị thể có 5 loại nút?"

---

## 1. Đồ thị dị thể (HIN) là gì?

Một đồ thị bình thường chỉ có 1 loại nút và 1 loại cạnh.
**Đồ thị dị thể (Heterogeneous Information Network — HIN)** có **nhiều loại nút** và **nhiều loại cạnh**, mỗi loại mang ngữ nghĩa khác nhau.

Trong bài này:

```
5 loại nút:    Post, User, Subreddit, Domain, Image
5 loại cạnh:   POSTED_BY, POSTED_IN, LINKS_TO, HAS_IMAGE, MEMBER_OF
```

Lý do chọn HIN thay vì đồ thị đồng nhất:
- Mỗi loại thực thể (người dùng, cộng đồng, domain tin tức, ảnh) có **chiều ngữ nghĩa riêng**
- GraphSAGE trên HIN có thể học message passing **khác nhau** cho từng loại quan hệ
- Cho phép tách bạch tín hiệu nhân quả (content, image) khỏi tín hiệu nhiễu (subreddit community)

---

## 2. Định nghĩa từng loại nút

### 2.1 Node Post (nút trung tâm cần phân loại)

| Thuộc tính | Giải thích |
|-----------|------------|
| `post_id` | ID bài post Reddit |
| `title` | Tiêu đề bài post (sau clean) |
| `score` | Điểm upvote tổng |
| `upvote_ratio` | Tỉ lệ upvote / tổng vote |
| `num_comments` | Số bình luận |
| `label_2way` | **Nhãn chính**: 0=Fake, 1=Real |
| `label_6way` | Nhãn chi tiết hơn (6 loại) |
| `split` | train / val / test |

**Features dùng trong GNN:** text embedding (768-d từ mpnet) + `[score, upvote_ratio, num_comments]` (3-d scalar)

### 2.2 Node User (người đăng bài)

| Thuộc tính | Giải thích |
|-----------|------------|
| `post_count` | Số bài đã đăng trong train |
| `avg_score` | Điểm trung bình |
| `avg_upvote_ratio` | Tỉ lệ upvote trung bình |
| `fake_rate` | % bài fake trong lịch sử train (1 − mean(label)) |

**Features dùng trong GNN:** vector 4 chiều trên

### 2.3 Node Subreddit (cộng đồng Reddit)

| Thuộc tính | Giải thích |
|-----------|------------|
| `post_count` | Số bài trong train |
| `fake_ratio_real` | Tỉ lệ bài fake trong train |
| `avg_score` | Điểm trung bình |

> **Vai trò nhân quả:** Subreddit là **confounder** (biến nhiễu) — nó vừa ảnh hưởng đến nội dung bài viết (một số subreddit chuyên đăng tin giả) lại vừa tương quan giả tạo với nhãn. Pipeline cắt cạnh Subreddit trong nhánh causal để triệt tiêu shortcut này.

### 2.4 Node Domain (tên miền nguồn tin)

| Thuộc tính | Giải thích |
|-----------|------------|
| `post_count` | Số bài liên kết đến domain này |
| `fake_ratio_real` | % bài fake từ domain này trong train |
| `avg_upvote_ratio` | Tỉ lệ upvote trung bình |
| `pagerank` | Điểm tầm quan trọng (tính bằng GDS PageRank) |

### 2.5 Node Image (ảnh minh họa)

| Thuộc tính | Giải thích |
|-----------|------------|
| `img_id` | ID ảnh (= `img_{post_id}`) |
| `post_id` | Post cha sở hữu ảnh |

**Features dùng trong GNN:** CLIP embedding 512 chiều (từ file `image_embeddings.npy`)

---

## 3. Định nghĩa từng loại cạnh

```
Post ──[POSTED_BY]──→ User           "bài này được đăng bởi user này"
Post ──[POSTED_IN]──→ Subreddit      "bài này thuộc cộng đồng này"
Post ──[LINKS_TO]───→ Domain         "bài này liên kết đến tên miền này"
Post ──[HAS_IMAGE]──→ Image          "bài này có ảnh đính kèm"
User ──[MEMBER_OF]──→ Subreddit      "user này tham gia cộng đồng này"
```

Cạnh không có: `CROSS_POST` (đã loại bỏ vì gây data leakage — cross-post xuất hiện trong cùng nhãn), `HAS_COMMENT` (không có dữ liệu comment text thực trong Fakeddit).

---

## 4. Quy trình chuyển đổi CSV → đồ thị (2 giai đoạn)

### Giai đoạn A: Đồ thị trong Neo4j (mục đích phân tích, dashboard)

**Công cụ:** Neo4j Graph Database + Neo4j GDS (Graph Data Science)

```
CSV files (posts.csv, users.csv, ...)
   │
   ▼  02_neo4j_import.py → MERGE (p:Post {post_id: r.post_id}) SET p += r
                         → MERGE (p)-[:POSTED_BY]->(u)  ...
   │
   ▼  Neo4j Property Graph (lưu trên disk, có thể query bằng Cypher)
   │
   ▼  GDS algorithms trên graph projection:
      - PageRank → ghi vào thuộc tính node Domain.pagerank
      - Louvain  → ghi vào Post.community_id
      - Betweenness → ghi vào User.betweenness
      - FastRP (64-d) → ghi vào Post.graph_embedding
```

Đây là đồ thị "thực" dùng để **khai phá dữ liệu, BI dashboard, visualization**. GNN không chạy trực tiếp trên Neo4j.

### Giai đoạn B: Đồ thị trong PyTorch Geometric (mục đích train GNN)

**Công cụ:** PyTorch Geometric (PyG) — thư viện `torch_geometric`

```
CSV files + .npy embeddings
   │
   ▼  05_train_gnn.py → build_heterodata()
   │
   ▼  HeteroData object (tensor format):
      data["Post"].x      = torch.Tensor (N_post, 771)   # text + scalar
      data["User"].x      = torch.Tensor (N_user, 4)
      data["Subreddit"].x = torch.Tensor (N_sub,  3)
      data["Domain"].x    = torch.Tensor (N_dom,  3)
      data["Image"].x     = torch.Tensor (N_img,  512)   # CLIP

      data["Post","posted_by","User"].edge_index = torch.Tensor (2, E1)
      data["Post","posted_in","Subreddit"].edge_index = ...
      ...
```

**Tại sao lại cần 2 giai đoạn?**

| Tiêu chí | Neo4j | PyG HeteroData |
|---------|-------|----------------|
| Mục đích | Lưu trữ, query, GDS algorithms | Tensor computation cho GNN |
| Truy vấn | Cypher query language | Python/PyTorch API |
| GDS algorithms | ✓ (PageRank, Louvain...) | ✗ |
| Gradient backprop | ✗ | ✓ |
| Tốc độ training | Chậm (disk I/O) | Nhanh (in-memory tensor) |

Neo4j làm nền tảng phân tích; PyG là nền tảng học.

---

## 5. Hình ảnh trực quan về đồ thị

```
[User: u/johndoe]
      |
      | MEMBER_OF
      |
      ▼
[Subreddit: r/news]
      ▲
      |
      | POSTED_IN
      |
[Post: "Scientists discover..."]  ──LINKS_TO──▶  [Domain: reuters.com]
      |
      | HAS_IMAGE
      |
      ▼
[Image: CLIP-512d vector]
      
[Post] ──POSTED_BY──▶ [User: u/johndoe]
```

Toàn bộ 6.200 bài post được kết nối theo cách này thành một đồ thị lớn với:
- ~6.200 Post nodes
- ~4.000–5.000 User nodes (unique authors)
- ~50–100 Subreddit nodes
- ~2.000–3.000 Domain nodes
- ~6.200 Image nodes

---

## Code tương ứng

```
pipeline/01_prepare_data.py
  create_neo4j_csvs()          → Tạo 5 file node CSV + 5 file edge CSV (line 281-488)

pipeline/02_neo4j_import.py
  import_to_neo4j()            → MERGE nodes/edges vào Neo4j (line 53-251)
  _run_gds()                   → Chạy PageRank, Louvain, FastRP (line 255-401)

pipeline/05_train_gnn.py
  build_heterodata()           → Chuyển CSV+npy → HeteroData tensor (line 332-506)
  T.ToUndirected()             → Tạo reverse edges tự động (line 504)
```
