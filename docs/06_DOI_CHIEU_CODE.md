# 06 — Đối chiếu Hình 2 với Mã nguồn thực tế

> **Câu hỏi thầy:** "Phải soi lại các bước trong sơ đồ xem có chuẩn theo code thực tế đã chạy hay chưa.
> Phải chú thích rõ ràng trong code: đoạn này dùng để làm gì, chỗ nào dùng thư viện, chỗ nào tự xây dựng."

---

## Bảng đối chiếu tổng thể

| Box trong Hình 2 | File | Hàm / Class | Dùng thư viện | Tự xây dựng |
|-----------------|------|-------------|---------------|-------------|
| **HeteroData Input** | 05_train_gnn.py | `build_heterodata()` | `torch_geometric.data.HeteroData`, `T.ToUndirected()` | Phần build edge index, feature concat |
| **G — Đồ thị đầy đủ** | 05_train_gnn.py | `CausalHeteroGNN.encode()` | `torch_geometric.nn.HeteroConv, SAGEConv` | Config edge types |
| **G_causal — Đồ thị can thiệp** | 05_train_gnn.py | `_cut_confounder_edges()` | Không | **Hoàn toàn tự xây dựng** (dict comprehension lọc edge type) |
| **Encoder GraphSAGE** | 05_train_gnn.py | `conv1, conv2 (HeteroConv)` | `SAGEConv` từ PyG | Cách tạo `make_conv_dict()` |
| **h_spurious** | 05_train_gnn.py | `spurious_head(h_post)` | `nn.Sequential` | MLP tự xây dựng |
| **h_causal** | 05_train_gnn.py | `causal_head(h_post_caus)` | `nn.Sequential` | MLP tự xây dựng |
| **GRL** | 05_train_gnn.py | `GRL`, `GradientReversal` | Không | **Hoàn toàn tự xây dựng** (`torch.autograd.Function`) |
| **Bộ phân loại Subreddit** | 05_train_gnn.py | `confounder_clf` | `nn.Sequential`, `nn.Linear` | Kết nối với GRL tự xây |
| **Bộ phân loại Fake/Real** | 05_train_gnn.py | `causal_clf_2way` | `nn.Sequential`, `nn.Linear` | MLP tự xây dựng |
| **L_ortho** | 05_train_gnn.py | training loop | `F.cosine_similarity` | Công thức tự viết |

---

## Chi tiết từng phần

### A. HeteroData Input — xây dựng graph tensor

**File:** `pipeline/05_train_gnn.py`, hàm `build_heterodata()` (line 332–506)

```python
# [THƯ VIỆN] HeteroData từ torch_geometric — container chuẩn cho heterogeneous graph
data = HeteroData()

# [TỰ XÂY] Concat feature thủ công theo từng node type
p_feats = np.concatenate([post_embeddings, p_scalar], axis=1)  # (N, 771)
data["Post"].x = torch.tensor(p_feats, dtype=torch.float)

# [THƯ VIỆN] build_edge dùng torch.tensor, nhưng logic map ID là code tự viết
data["Post","posted_by","User"].edge_index = build_edge(...)

# [THƯ VIỆN] ToUndirected tự động tạo reverse edge (User→Post từ Post→User)
data = T.ToUndirected()(data)
```

**Khớp Hình 2:** ✓ Box "HeteroData Input" là kết quả của hàm này. Tất cả 5 node types và 5 edge types đúng như sơ đồ.

---

### B. Hai nhánh đồ thị (G và G_causal)

**File:** `pipeline/05_train_gnn.py`, `CausalHeteroGNN.forward()` (line 290–327)

```python
# [TỰ XÂY] Nhánh G đầy đủ — không cắt gì cả
h_full = self.encode(x_dict, edge_index_dict)

# [TỰ XÂY HOÀN TOÀN] Nhánh G_causal — cắt cạnh Subreddit
# _cut_confounder_edges là dict comprehension tự viết, không dùng thư viện
h_caus = self.encode(x_dict, self._cut_confounder_edges(edge_index_dict))
```

```python
@staticmethod
def _cut_confounder_edges(edge_index_dict):
    # [TỰ XÂY] Logic cắt: giữ cạnh nếu source VÀ destination đều KHÔNG phải Subreddit
    return {k: v for k, v in edge_index_dict.items()
            if k[0] != "Subreddit" and k[2] != "Subreddit"}
```

**Khớp Hình 2:** ✓ Box "G (đồ thị đầy đủ)" và "G_causal (xóa cạnh Subreddit)" trong sơ đồ khớp chính xác.

---

### C. Heterogeneous GraphSAGE Encoder (dùng chung trọng số)

**File:** `pipeline/05_train_gnn.py`, class `CausalHeteroGNN` (line 124–327)

```python
# [THƯ VIỆN] SAGEConv từ torch_geometric.nn — GraphSAGE convolution chuẩn
# [THƯ VIỆN] HeteroConv — wrapper cho heterogeneous graph, dispatch đúng SAGEConv theo edge type
def make_conv_dict():
    return {edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in metadata[1]}

self.conv1 = HeteroConv(make_conv_dict(), aggr="sum")  # lớp GraphSAGE 1
self.conv2 = HeteroConv(make_conv_dict(), aggr="sum")  # lớp GraphSAGE 2

# [TỰ XÂY] Hàm encode gọi conv1 và conv2 theo thứ tự với dropout
def encode(self, x_dict, edge_index_dict):
    h0 = {k: self.dropout(F.relu(self.proj[k](v)))    # projection
          for k, v in x_dict.items() if k in self.proj}
    h  = self.conv1(h0, edge_index_dict)               # lớp 1
    h  = {k: self.dropout(F.relu(v)) for k, v in h.items()}
    h  = self.conv2(h, edge_index_dict)                # lớp 2
    return h
```

**Khớp Hình 2:** ✓ Box "Heterogeneous GraphSAGE Encoder (dùng chung trọng số, d=96)" — cả G và G_causal đều gọi cùng `self.conv1` và `self.conv2` (cùng object trọng số).

---

### D. Gradient Reversal Layer (GRL)

**File:** `pipeline/05_train_gnn.py`, class `GradientReversal` và `GRL` (line 102–119)

```python
# [TỰ XÂY HOÀN TOÀN] GRL không có trong bất kỳ thư viện nào
# Dùng torch.autograd.Function để hook vào backward pass
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)           # forward: identity (pass-through)

    @staticmethod
    def backward(ctx, grad_output):
        # backward: đảo dấu gradient × α — đây là "thủ thuật" chính
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)  # gọi hàm custom autograd
```

**Dùng trong forward:**
```python
# h_causal đi qua GRL trước khi vào Confounder Classifier
sub_pred_causal = self.confounder_clf(self.grl(h_c))
```

**Khớp Hình 2:** ✓ Box "GRL (α=2.0)" — α=2.0 được set ở config (`GRL_ALPHA = 2.0`).

---

### E. Các bộ phân loại (Classifiers)

**File:** `pipeline/05_train_gnn.py`, khởi tạo trong `__init__` (line 186–216)

```python
# [THƯ VIỆN] nn.Sequential, nn.Linear, nn.Dropout, nn.ReLU từ PyTorch
# [TỰ XÂY] Cách kết nối (kiến trúc MLP) và việc có 4 classifier song song
def mlp(in_dim, out_dim, dropout_p=dropout):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),   # fully-connected layer 1
        nn.ReLU(),                    # activation
        nn.Dropout(p=dropout_p),      # regularization
        nn.Linear(in_dim, out_dim),   # fully-connected layer 2 → đầu ra
    )

# 4 classifiers:
self.baseline_clf_2way = mlp(hidden_channels, 2)    # dùng h_post từ G đầy đủ
self.causal_clf_2way   = mlp(hidden_channels, 2)    # ← CHÍNH: dùng h_causal
self.baseline_clf_6way = mlp(hidden_channels, 6)
self.causal_clf_6way   = mlp(hidden_channels, 6)

# Confounder classifier (adversarial):
self.confounder_clf = nn.Sequential(
    nn.Linear(hidden_channels, hidden_channels // 2),
    nn.ReLU(),
    nn.Dropout(p=dropout),
    nn.Linear(hidden_channels // 2, num_subreddits),  # dự đoán subreddit ID
)
```

**Khớp Hình 2:** ✓ Hai box phía dưới:
- "Bộ phân loại Subreddit (Confounder)" = `confounder_clf`
- "Bộ phân loại Fake / Real (Đầu ra chính → ŷ)" = `causal_clf_2way`

---

### F. Hàm Loss tổng hợp

**File:** `pipeline/05_train_gnn.py`, training loop (line 755–795)

```python
# [THƯ VIỆN] F.cross_entropy từ PyTorch
# [TỰ XÂY] Cách kết hợp nhiều loss với hệ số trọng số

loss = (
    loss_base_2   + 0.5 * loss_base_6       # Baseline: Fake/Real + 6-way
  + loss_causal_2 + 0.5 * loss_causal_6     # Causal: Fake/Real + 6-way
  + 0.5 * loss_sub_spurious                  # Confounder: h_spurious → subreddit
  + W_SUB_ADV * loss_sub_adv                 # Adversarial: h_causal → subreddit (L_adv)
  + 0.2 * loss_ortho                         # Trực giao: h_causal ⊥ h_spurious
)
```

**Khớp Hình 2:** ✓ 3 nhãn loss trong sơ đồ:
- `L_spurious` = `loss_sub_spurious`
- `L_adv` = `loss_sub_adv` (qua GRL)
- `L_causal` = `loss_causal_2 + 0.5*loss_causal_6`

---

### G. Evaluation — Inference trên test set

**File:** `pipeline/06_evaluate.py`, hàm `main()` (line 67–461)

```python
# [THƯ VIỆN] torch.no_grad() — tắt gradient để tiết kiệm bộ nhớ
# [TỰ XÂY] mask_post_edges — tự xây dựng để loại bỏ cạnh test Post (inductive eval)
# [THƯ VIỆN] sklearn.metrics — accuracy_score, f1_score, roc_auc_score

# Inductive evaluation: xóa cạnh graph của test posts
inductive_edge_dict = mask_post_edges(data.edge_index_dict, all_test_ids_set)
with torch.no_grad():
    (_, ind_out_causal_2, ...) = model(data.x_dict, inductive_edge_dict)

# Tính metrics
metrics_causal_2way = compute_full_metrics(y_test_2way, ind_out_causal_2[test_mask], ...)
```

**Khớp Hình 2:** ✓ — Bộ phân loại chính (causal_clf_2way) nhận đầu ra từ h_causal của G_causal, cho ra logits Fake/Real.

---

## Kiểm tra: Điều gì đúng, điều gì cần lưu ý

| Điểm | Nhận xét |
|------|----------|
| ✓ GRL được tự xây dựng đúng cách | `torch.autograd.Function` là phương pháp chuẩn |
| ✓ `_cut_confounder_edges` cắt đúng theo tên node type | Không hardcode index |
| ✓ Hai nhánh G và G_causal dùng CHUNG `conv1`, `conv2` | Khớp mô tả "shared encoder" trong Hình 2 |
| ⚠ Comment trong code: một số hàm đã có nhưng chưa đầy đủ | → Đã bổ sung comment ở bước tiếp theo |
| ✓ Counterfactual và causal path attribution trong 06_evaluate.py | Khớp với "thí nghiệm nhân quả" mô tả trong bài |

---

## Sơ đồ Hình 2 ↔ Code mapping tóm tắt

```
Hình 2                               Code (05_train_gnn.py)
─────────────────────────────────────────────────────────
HeteroData Input              ←→  build_heterodata() (line 332)
G đầy đủ                      ←→  encode(x_dict, edge_index_dict) (line 243)
G_causal                      ←→  encode(x_dict, _cut_confounder_edges(...)) (line 304)
GraphSAGE Encoder             ←→  HeteroConv(SAGEConv, ...) conv1 + conv2 (line 183)
h_spurious                    ←→  spurious_head(h_post) (line 311)
h_causal                      ←→  causal_head(h_post_caus) (line 311)
GRL (α=2.0)                   ←→  GRL(alpha=2.0) (line 205)
Bộ phân loại Subreddit        ←→  confounder_clf (line 199)
Bộ phân loại Fake/Real (ŷ)    ←→  causal_clf_2way (line 212) ← ĐẦU RA CHÍNH
L_spurious                    ←→  loss_sub_spurious (line 777)
L_adv                         ←→  loss_sub_adv (line 778)
L_causal                      ←→  loss_causal_2 + loss_causal_6 (line 770-772)
L_ortho                       ←→  loss_ortho (line 781-783)
```
