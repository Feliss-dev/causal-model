# 03 — Tại sao tách hai đồ thị (G và G_causal) rồi lại hòa nhập?

> **Câu hỏi thầy:** "Cần giải thích tại sao lại tách thành hai phiên bản đồ thị,
> sau đó tại sao lại có bước hòa nhập hoặc đối sánh chúng."

---

## 1. Vấn đề cốt lõi: Shortcut Learning (học tắt)

Bài toán phát hiện tin giả trên Reddit có một **cạm bẫy đặc thù**:

> Một số subreddit **chuyên đăng tin giả** (r/TheOnion) hoặc **chuyên đăng tin thật** (r/worldnews).
> Nếu model học được quy luật "*bài từ r/TheOnion → Fake*", nó đạt accuracy cao trên train
> **mà không cần đọc nội dung bài viết**. Đây gọi là "shortcut" hay "spurious correlation" (tương quan giả tạo).

**Hậu quả:** Khi gặp subreddit mới chưa từng thấy (OOD), model bị mất hoàn toàn — vì shortcut không còn dùng được.

---

## 2. Ngôn ngữ nhân quả (Causal DAG)

Trong lý thuyết nhân quả (Pearl, 2009), mô hình được biểu diễn qua **Causal DAG**:

```
Subreddit (C1) ──→ Content (X) ──→ Label (Y = Fake/Real)
     │                                     ▲
     └────────────────────────────────────→┘
              "backdoor path" (con đường tắt)

Domain (C2) ──→ Label (Y)
Image  (I)  ──→ Label (Y)
```

**Mũi tên nhân quả thực:** Content → Y, Image → Y, Domain → Y

**Backdoor path (đường nhiễu):** Subreddit → Content → Y và Subreddit → Y trực tiếp

Một model học trên full graph học **cả hai** — cả tín hiệu nhân quả lẫn tín hiệu nhiễu.

---

## 3. Giải pháp: Backdoor Adjustment (Can thiệp do-calculus)

Lý thuyết Pearl cho rằng ta có thể triệt tiêu backdoor path bằng cách **can thiệp (do-calculus)**:

> **do(Subreddit = neutral)**: "Giả sử không có thông tin nào từ community cụ thể,
> model phân loại dựa trên gì?"

Thực hiện can thiệp này trong đồ thị bằng cách **xóa tất cả cạnh liên quan đến Subreddit** trong một bản sao của đồ thị:

```
G (đồ thị đầy đủ):
  Post ──POSTED_IN──→ Subreddit
  User ──MEMBER_OF──→ Subreddit
  ← Subreddit gửi message đến Post và User qua message passing →

G_causal (đồ thị can thiệp):
  [XÓA mọi cạnh incident on Subreddit]
  Post ──POSTED_BY──→ User
  Post ──LINKS_TO───→ Domain
  Post ──HAS_IMAGE──→ Image
  ← Subreddit bị cô lập, KHÔNG gửi message cho ai →
```

**Code thực hiện (05_train_gnn.py line 231-234):**
```python
@staticmethod
def _cut_confounder_edges(edge_index_dict):
    """Remove every edge incident on the Subreddit node (backdoor cut)."""
    return {k: v for k, v in edge_index_dict.items()
            if k[0] != "Subreddit" and k[2] != "Subreddit"}
```

---

## 4. Tại sao cần TẠO cả hai — không bỏ hẳn G đầy đủ?

Thiết kế **song song hai nhánh** (G + G_causal) cho phép học **hai loại biểu diễn**:

| Nhánh | Đồ thị | Biểu diễn | Vai trò |
|-------|--------|-----------|---------|
| Spurious | G đầy đủ | `h_spurious` | Nắm bắt cả nhân quả + nhiễu. Dùng để học Confounder Classifier (đoán subreddit ID) — mục đích là có "bạn tập võ" cho adversarial training |
| Causal | G_causal | `h_causal` | Chỉ nắm bắt tín hiệu nhân quả. Đây là biểu diễn phân loại Fake/Real cuối cùng |

Nếu chỉ giữ G_causal và bỏ hoàn toàn G, ta mất đi khả năng **đối kháng học** (adversarial training) qua GRL — không có gì để model "kháng cự".

---

## 5. Hòa nhập: Gradient Reversal Layer (GRL)

Sau khi có `h_causal`, pipeline sử dụng **GRL (Gradient Reversal Layer)** để ép `h_causal` KHÔNG chứa thông tin về subreddit:

```
h_causal
   │
   ▼ GRL(α=2.0)  ← trong forward pass: pass-through
                  ← trong backward pass: ĐẢO DẤU gradient, nhân với α
   │
   ▼ Confounder Classifier (dự đoán subreddit ID từ h_causal)
   │
   ▼ L_adv = CrossEntropy(pred_subreddit, true_subreddit)
```

**Cơ chế hoạt động của GRL:**

```
[Mục tiêu Confounder Classifier]:  cố gắng đoán đúng subreddit
     → gradient gradient tốt lên Confounder Classifier

[Qua GRL, gradient bị ĐẢO DẤU]:
     → Encoder (h_causal) nhận gradient ngược
     → Encoder bị ép học cách XÓA tín hiệu subreddit khỏi h_causal
```

**Toán học:**
```
GRL forward:  f(x) = x
GRL backward: ∂L/∂x → −α · ∂L/∂x  (đảo dấu × α)

Điều này tương đương minimax:
  max_θ_enc  min_θ_clf  [L_task − α · L_adv]
```

---

## 6. Ràng buộc trực giao (L_ortho)

Ngoài GRL, pipeline còn thêm **L_ortho** để ép `h_causal` và `h_spurious` KHÔNG overlap:

```python
loss_ortho = torch.mean(torch.abs(
    F.cosine_similarity(h_c[train_mask], h_s[train_mask])
))
```

Nếu cosine similarity giữa hai vector gần 1, chúng đang học cùng thông tin — lãng phí.
L_ortho phạt sự trùng lặp này.

---

## 7. Sơ đồ luồng đầy đủ

```
HeteroData Input
        │
        ├─────────────────────────────────────────┐
        ▼                                         ▼
  G (đầy đủ)                             G_causal (cắt Subreddit)
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
          GraphSAGE Encoder (DÙNG CHUNG TRỌNG SỐ)
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
      h_post (từ G)           h_post_caus (từ G_causal)
          │                         │
          ▼                         ▼
    spurious_head()           causal_head()
          │                         │
          ▼                         ├────────────────────────┐
    h_spurious                 h_causal                     │
          │                         │                        ▼
          ▼                         ▼                     GRL(α=2.0)
   Confounder                 Fake/Real                      │
   Classifier                Classifier ← ŷ chính           ▼
   L_spurious                L_causal             Confounder Classifier
                                                   L_adv (adversarial)
          ↕ L_ortho (trực giao) ↕
```

**Tổng loss:**
```
L = L_base_2way + 0.5·L_base_6way
  + L_causal_2way + 0.5·L_causal_6way   ← nhiệm vụ phân loại chính
  + 0.5·L_spurious + 2.0·L_adv          ← disentanglement adversarial
  + 0.2·L_ortho                          ← trực giao
```

---

## Code tương ứng

```
pipeline/05_train_gnn.py

CausalHeteroGNN.forward() (line 290-327):
  h_full = encode(x_dict, edge_index_dict)          # Nhánh G đầy đủ
  h_caus = encode(x_dict, _cut_confounder_edges(...))  # Nhánh G_causal
  h_c = causal_head(h_post_caus)     # biểu diễn nhân quả
  h_s = spurious_head(h_post)        # biểu diễn phụ
  pred_causal_2 = causal_clf_2way(h_c)   # ← kết quả phân loại chính
  sub_pred_causal = confounder_clf(grl(h_c))  # adversarial head

Training loop (line 743-860):
  loss_sub_adv = F.cross_entropy(sub_pred_causal[mask], sub_labels)  # L_adv
  loss_ortho = cosine_similarity(h_c, h_s).mean().abs()              # L_ortho
```
