# 04 — Bộ phân loại (Classifier): Hình thù, Toán học và Cơ chế ra quyết định

> **Câu hỏi thầy (nhắc rất kỹ):** "Hình thù mặt mũi của bộ phân loại là gì?
> Nó được biểu diễn như một hàm toán học, một cấu trúc mô hình hay một đồ thị?
> Nó dùng để kiểm thử test set như thế nào và kết luận Fake/Real dựa trên cơ sở nào?"

---

## Câu trả lời trực tiếp

**Bộ phân loại là một MLP (Multi-Layer Perceptron) — không phải đồ thị, không phải Neo4j object.**

Nó nhận vector embedding `h_causal` (96 chiều) của một Post và trả về 2 số (logits).
Số nào lớn hơn → Post được phân loại theo nhãn đó.

---

## 1. Cấu trúc chính xác của Classifier

Trong code (`05_train_gnn.py`, line 187-216), bộ phân loại được định nghĩa qua hàm `mlp()`:

```python
def mlp(in_dim, out_dim, dropout_p=dropout):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),   # Linear: (96, 96)
        nn.ReLU(),                    # Kích hoạt phi tuyến
        nn.Dropout(p=dropout_p),      # Dropout = 0.4 (tránh overfit)
        nn.Linear(in_dim, out_dim),   # Linear: (96, 2) cho 2-way
    )

self.causal_clf_2way   = mlp(hidden_channels, 2)   # Fake/Real
self.baseline_clf_2way = mlp(hidden_channels, 2)   # Fake/Real (baseline)
self.causal_clf_6way   = mlp(hidden_channels, 6)   # 6 loại
self.baseline_clf_6way = mlp(hidden_channels, 6)
```

**Vậy có 4 bộ phân loại, nhưng bộ CHÍNH là `causal_clf_2way`.**

---

## 2. Biểu diễn toán học

### Đầu vào
```
h_causal ∈ ℝ^96  (vector 96 chiều — output của causal_head)
```

### Hai lớp Linear + ReLU + Dropout

```
Bước 1 (Linear):  z₁ = W₁ · h_causal + b₁       W₁ ∈ ℝ^(96×96)
Bước 2 (ReLU):    a₁ = max(0, z₁)
Bước 3 (Dropout): a₁' = a₁ ⊙ mask   (mask_i ~ Bernoulli(0.6))
Bước 4 (Linear):  logits = W₂ · a₁' + b₂         W₂ ∈ ℝ^(96×2)

Kết quả: logits ∈ ℝ²  = [logit_Fake, logit_Real]
```

### Ra quyết định (dự đoán)

```
Xác suất:  p = Softmax(logits)  =  [p(Fake), p(Real)]
           p(Real) = exp(logit_Real) / (exp(logit_Fake) + exp(logit_Real))

Nhãn dự đoán:  ŷ = argmax(logits) = argmax([logit_Fake, logit_Real])
                                   = 0 nếu logit_Fake > logit_Real  → Fake
                                   = 1 nếu logit_Real > logit_Fake  → Real
```

**Ví dụ số:**
```
h_causal = [0.2, -0.5, 0.8, ..., 0.1]  (96 chiều)
logits = [-1.3,  2.7]  (2 chiều: [Fake_score, Real_score])
Softmax → [0.027, 0.973]  → p(Real) = 97.3%
argmax  → 1 → KẾT LUẬN: Real
```

---

## 3. Phân loại 6-way (chi tiết hơn)

Tương tự nhưng đầu ra có 6 chiều:

```
logits ∈ ℝ^6 = [score_True, score_Satire, score_Misleading,
                score_Imposter, score_FalseConn, score_Manipulated]
ŷ = argmax(logits)  → 1 trong 6 loại tin giả
```

---

## 4. Toàn bộ pipeline từ Post → Fake/Real

```
Post (bài post Reddit)
   │
   ▼ BƯỚC 1: Feature extraction (01_prepare_data.py)
   │  text → SentenceTransformer → embedding 768-d
   │  ảnh  → CLIP ViT-B/32 → embedding 512-d (lưu Image node)
   │  scalar → [score, upvote_ratio, num_comments]
   │
   ▼ BƯỚC 2: Node feature construction (05_train_gnn.py build_heterodata)
   │  Post.x = concat([text_768d, scalar_3d]) = tensor (771,)
   │
   ▼ BƯỚC 3: Linear projection (CausalHeteroGNN.proj)
   │  h₀ = ReLU(Linear(771, 96)(Post.x))  → (96,)
   │
   ▼ BƯỚC 4: GraphSAGE message passing, 2 lớp (conv1, conv2)
   │  Lớp 1: h₁[Post] = SAGEConv(h₀[Post], messages từ neighbors)  → (96,)
   │  Lớp 2: h₂[Post] = SAGEConv(h₁[Post], messages từ neighbors)  → (96,)
   │  (Nhánh causal: messages từ Subreddit bị chặn hoàn toàn)
   │
   ▼ BƯỚC 5: causal_head MLP
   │  h_causal = ReLU(Linear(96,96)(h₂)) + Dropout → (96,)
   │
   ▼ BƯỚC 6: causal_clf_2way — Bộ phân loại
   │  logits = Linear(96, 2)(ReLU(Linear(96,96)(h_causal)))  → (2,)
   │
   ▼ BƯỚC 7: Ra quyết định
      ŷ = argmax(logits)
      ŷ = 0 → Fake   |   ŷ = 1 → Real
```

---

## 5. Cơ chế kiểm thử tập test

Trong quá trình **test**, không có gradient — chỉ forward pass:

```python
model.eval()         # tắt Dropout và BatchNorm
with torch.no_grad():
    (_, pred_causal_2, ...) = model(data.x_dict, data.edge_index_dict)

# pred_causal_2 có shape (N_all_posts, 2)
# Chỉ lấy kết quả tại test posts:
test_logits = pred_causal_2[test_mask]         # (N_test, 2)
y_pred = torch.argmax(test_logits, dim=1)       # (N_test,)  — 0 hoặc 1
```

**Chế độ inductive (nghiêm ngặt hơn):**
Trước khi forward pass, tất cả cạnh của test Post được xóa khỏi graph.
Test Post chỉ được phân loại từ feature riêng của nó, không nhận message từ neighbor.
Đây là chế độ thực tế nhất — tương ứng với "phân loại bài mới chưa từng thấy".

---

## 6. Cơ sở quyết định Fake/Real

Kết luận dựa trên **tổng hợp 5 loại tín hiệu** được truyền qua message passing:

| Tín hiệu | Nguồn | Chiều |
|---------|-------|-------|
| Ngữ nghĩa tiêu đề | SentenceTransformer (Post.x) | 768-d |
| Thống kê bài post | score, upvote_ratio, num_comments | 3-d |
| Lịch sử người dùng | fake_rate, avg_score... (User.x) | 4-d |
| Uy tín nguồn tin | fake_ratio_real (Domain.x) | 3-d |
| Tín hiệu ảnh | CLIP embedding (Image.x) | 512-d |

Subreddit **không được** dùng trong nhánh causal (bị cắt — xem file 03).

Sau 2 lớp GraphSAGE, tất cả tín hiệu này được **hội tụ** vào vector 96 chiều `h_causal`.
Bộ phân loại MLP học cách ánh xạ vector này sang `{Fake, Real}`.

---

## 7. Tóm tắt cho buổi báo cáo

| Câu hỏi | Câu trả lời |
|---------|-------------|
| Bộ phân loại là gì? | MLP 2 lớp: Linear(96→96) + ReLU + Dropout + Linear(96→2) |
| Biểu diễn như gì? | Hàm toán học f: ℝ^96 → ℝ^2, không phải đồ thị hay Neo4j |
| Đầu vào là gì? | h_causal — embedding 96-d của Post sau GraphSAGE causal branch |
| Đầu ra là gì? | 2 logits (raw scores) cho [Fake, Real] |
| Ra quyết định thế nào? | argmax(logits) → 0=Fake, 1=Real |
| Cơ sở quyết định? | Tổng hợp tín hiệu text + scalar + user history + domain credibility + image |

---

## Code tham chiếu

```python
# 05_train_gnn.py — line 186-216
def mlp(in_dim, out_dim, dropout_p=dropout):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(in_dim, out_dim),
    )
self.causal_clf_2way = mlp(hidden_channels, 2)   # ← ĐÂY LÀ BỘ PHÂN LOẠI CHÍNH

# forward() — line 317
pred_causal_2 = self.causal_clf_2way(h_c)   # h_c ∈ ℝ^96 → logits ∈ ℝ²

# Training — line 770
loss_causal_2 = F.cross_entropy(pred_causal_2[train_mask], y_train_2way)

# Testing — line 906
m_causal_2 = compute_full_metrics(y_test_2way, pred_causal_2[test_mask], label_2way)
```
