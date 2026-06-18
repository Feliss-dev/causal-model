# Bộ tài liệu hướng dẫn giải trình — CausalHeteroGNN

> **Mục đích:** Trả lời trực tiếp từng câu hỏi của thầy phản biện về Hình 2,
> bộ phân loại (classifier), và sự khớp giữa sơ đồ với mã nguồn thực tế.

---

## Cấu trúc tài liệu

| File | Nội dung | Câu hỏi thầy |
|------|----------|--------------|
| [01_DU_LIEU_DAU_VAO.md](01_DU_LIEU_DAU_VAO.md) | Dữ liệu Fakeddit → CSV được xây dựng như thế nào | Câu 1 — "Dữ liệu đầu vào" |
| [02_XAY_DUNG_DO_THI.md](02_XAY_DUNG_DO_THI.md) | 5 loại nút, 5 loại cạnh, công cụ dùng (Neo4j, PyG) | Câu 1 — "Xây dựng đồ thị" |
| [03_HAI_NHANH_DO_THI.md](03_HAI_NHANH_DO_THI.md) | Tại sao tách G / G\_causal, tại sao hòa nhập | Câu 1 — "Tách biệt và hòa nhập" |
| [04_BO_PHAN_LOAI.md](04_BO_PHAN_LOAI.md) | Bộ phân loại là gì, biểu diễn toán học, cơ chế ra kết quả | Câu 2 — "Bộ phân loại" |
| [05_VI_DU_MINH_HOA.md](05_VI_DU_MINH_HOA.md) | Ví dụ 1 bài post thực tế đi qua toàn hệ thống | Câu 3 — "Ví dụ cụ thể" |
| [06_DOI_CHIEU_CODE.md](06_DOI_CHIEU_CODE.md) | Bản đồ: mỗi ô Hình 2 ↔ đoạn code cụ thể | Câu 4 — "Đối chiếu code" |
| [07_HUONG_DAN_CHAY_CODE.md](07_HUONG_DAN_CHAY_CODE.md) | Hướng dẫn chạy 4 bước pipeline, lệnh, thời gian, lỗi thường gặp | Chạy thực tế |

---

## Luồng code 4 bước

```
01_prepare_data.py
  ↓  Đọc TSV thô → lọc, sample, tải ảnh, tạo embedding, xuất CSV

02_neo4j_import.py
  ↓  Nạp CSV vào Neo4j → chạy GDS (PageRank, Louvain, Betweenness, FastRP)

05_train_gnn.py
  ↓  Đọc CSV/embedding → HeteroData → CausalHeteroGNN → train → lưu model

06_evaluate.py
     Load model → inductive eval → OOD F1-drop → counterfactual → giải thích
```

---

## Sơ đồ Hình 2 — tóm tắt nhanh

```
HeteroData Input  (Post, User, Subreddit, Domain, Image)
        │
        ├──────────────────────────────┐
        ▼                              ▼
   G (đồ thị đầy đủ)         G_causal (cắt cạnh Subreddit)
        │                              │
        └──────────┬───────────────────┘
                   ▼
      Heterogeneous GraphSAGE Encoder  (chia sẻ trọng số)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  h_spurious           h_causal
  (biểu diễn phụ)      (biểu diễn nhân quả)
        │                     │
        │          GRL ────── h_causal
        ▼                     │
  Bộ phân loại         Bộ phân loại
  Subreddit            Fake / Real  ← đầu ra chính ŷ
  (L_spurious)         (L_causal)
```

---

## Cách đọc tài liệu nhanh nhất

Nếu thầy hỏi về **sơ đồ kiến trúc** → đọc file 01, 02, 03 theo thứ tự.

Nếu thầy hỏi **bộ phân loại là gì** → đọc file 04 trực tiếp.

Nếu cần **ví dụ cụ thể** → file 05.

Nếu thầy hỏi **code khớp sơ đồ không** → file 06.
