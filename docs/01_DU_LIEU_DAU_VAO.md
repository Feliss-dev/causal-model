# 01 — Dữ liệu đầu vào: Từ Fakeddit thô đến tập train/val/test

> **Câu hỏi thầy:** "Dữ liệu huấn luyện được triển khai như thế nào từ nguồn dữ liệu ban đầu?"

---

## 1. Nguồn dữ liệu gốc

Fakeddit là bộ dữ liệu Reddit multimodal do Nakamura et al. (2020) thu thập.
Có 3 file TSV được cung cấp:

| File | Vai trò |
|------|---------|
| `multimodal_train.tsv` | ~614 k bài post, có nhãn 2-way và 6-way |
| `multimodal_validate.tsv` | ~10 k bài, dùng để chọn val và seen-test |
| `multimodal_test_public.tsv` | File test chính thức (tùy chọn) |

Mỗi dòng trong TSV là một bài post Reddit với các cột chính:

```
id | title | clean_title | score | upvote_ratio | num_comments |
author | subreddit | domain | image_url | hasImage |
2_way_label | 6_way_label
```

- **`2_way_label`**: `1` = Real (tin thật), `0` = Fake (tin giả)
- **`6_way_label`**: 0=True, 1=Satire, 2=Misleading, 3=Imposter, 4=FalseConnection, 5=Manipulated

---

## 2. Lý do phải sampling và lọc

Bộ gốc Fakeddit không cân bằng (nhiều bài Real hơn Fake) và có nhiều bài không có ảnh.
Pipeline thực hiện 3 bước lọc theo thứ tự:

```
TSV thô (~614k dòng)
   │
   ▼ Bước 1: Lọc bài CÓ ẢNH (hasImage=True và image_url bắt đầu bằng "http")
   │
   ▼ Bước 2: Tách OOD subreddits ra khỏi pool train/val
   │          OOD = {neutralnews (Real), theonion (Fake)}
   │          → Hai community này chỉ xuất hiện trong OOD test, KHÔNG dùng để train
   │
   ▼ Bước 3: Balanced sampling (lấy đều 2 class)
              train:     2.500 Real + 2.500 Fake = 5.000 bài
              val:         200 Real +   200 Fake =   400 bài
              seen-test:   100 Real +   100 Fake =   200 bài
              OOD-test:    150 neutralnews + 150 theonion = 300 bài
```

**Tại sao phải cân bằng?**
Nếu không cân bằng, model chỉ cần đoán "Real" liên tục vẫn đạt accuracy cao — không học được gì.

**Tại sao tách OOD?**
Để kiểm tra khả năng *tổng quát hóa* (generalization): model có phân loại được bài từ các cộng đồng chưa từng thấy không?
Một model "học tắt" (memorize community style) sẽ thất bại trên OOD.

---

## 3. Tải ảnh và trích xuất embedding

Sau khi chọn được danh sách bài, pipeline tải ảnh song song:

```
Bài post có image_url
   │
   ▼ download_images_parallel() — 15 luồng song song
   │  Lưu tại: data/images/{post_id}.jpg
   │
   ▼ extract_clip_embeddings()
      Dùng: openai/clip-vit-base-patch32 (thư viện: transformers, PIL, torch)
      Kết quả: mảng numpy (N, 512) — mỗi ảnh thành vector 512 chiều L2-normalized
      Lưu: data/processed/image_embeddings.npy
```

Song song đó, tiêu đề bài post được encode bằng SentenceTransformer:

```
title (text)
   │
   ▼ SentenceTransformer("all-mpnet-base-v2")
      Kết quả: mảng numpy (N, 768) — mỗi tiêu đề thành vector 768 chiều
      Lưu: data/processed/post_embeddings.npy
```

**Tại sao dùng CLIP + SentenceTransformer?**
- CLIP nắm bắt ngữ nghĩa ảnh (visual semantics), đã được pretrain trên 400M cặp ảnh-text
- all-mpnet-base-v2 tốt hơn BERT gốc cho semantic similarity (benchmark BEIR)
- Cả hai đều là feature đã học sẵn (pretrained), không cần train lại

---

## 4. Tính feature thống kê (train-only, tránh data leakage)

Mỗi User, Subreddit, Domain không chỉ có ID mà còn có các **feature thống kê** được tính TỪ DỮ LIỆU TRAIN:

| Node | Features | Cách tính |
|------|----------|-----------|
| User | `post_count`, `avg_score`, `avg_upvote_ratio`, `fake_rate` | Groupby `author` trên train split |
| Subreddit | `post_count`, `fake_ratio_real`, `avg_score` | Groupby `subreddit` trên train split |
| Domain | `post_count`, `fake_ratio_real`, `avg_upvote_ratio` | Groupby `domain` trên train split |

**Tại sao chỉ dùng train?**
Nếu dùng cả val/test để tính (ví dụ fake_rate của User), model sẽ gián tiếp "nhìn thấy" nhãn của tập test trước khi phân loại — đây là data leakage. Pipeline gán giá trị mặc định trung tính (0.5) cho User/Subreddit/Domain chỉ xuất hiện ở val/test.

---

## 5. Kết quả đầu ra của bước 01

Sau khi chạy `01_prepare_data.py`, thư mục `data/processed/` có:

```
data/processed/
├── sampled_master.csv        ← bảng tổng hợp tất cả split
├── posts.csv                 ← node Post (6.200 dòng)
├── users.csv                 ← node User (unique authors)
├── subreddits.csv            ← node Subreddit
├── domains.csv               ← node Domain
├── images.csv                ← node Image (metadata)
├── posted_by.csv             ← cạnh Post→User
├── posted_in.csv             ← cạnh Post→Subreddit
├── links_to.csv              ← cạnh Post→Domain
├── has_image.csv             ← cạnh Post→Image
├── member_of.csv             ← cạnh User→Subreddit
├── post_embeddings.npy       ← (N, 768) text features
└── image_embeddings.npy      ← (N, 512) CLIP image features
```

> **Điểm then chốt:** Tất cả dữ liệu ở bước này vẫn ở dạng bảng phẳng (CSV).
> Việc kết nối chúng thành **đồ thị** xảy ra ở bước 02 (Neo4j) và bước 05 (PyG HeteroData).

---

## Code tương ứng

```
pipeline/01_prepare_data.py
  filter_with_images()         → Lọc bài có ảnh (line 262-267)
  balanced_sample()            → Sample cân bằng 2 class (line 270-276)
  download_images_parallel()   → Tải ảnh đa luồng (line 105-118)
  extract_clip_embeddings()    → CLIP embedding 512-d (line 123-183)
  compute_real_features()      → Tính stats từ train-only (line 188-257)
  create_neo4j_csvs()          → Xuất CSV nodes + edges (line 281-488)
```
