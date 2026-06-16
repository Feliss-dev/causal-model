"""
05_train_gnn.py
===============
Train Causal Heterogeneous GNN trên Fakeddit HIN.

Thay đổi so với phiên bản cũ:
- Xoá Comment node và CROSS_POST edge khỏi model/graph
- User features: [post_count, avg_score, avg_upvote_ratio, betweenness] (4 thực)
  → nn.Linear(4, hidden) thay vì (4 mock)
- Thêm gradient clipping (max_norm=1.0) → ngăn gradient explosion
- Thêm class weights cho 6-way loss (xử lý class imbalance)
- Tính Label Flip Rate (LFR) sau counterfactual
- Ghi chú rõ transductive setting

Lưu ý về transductive evaluation:
  GNN dùng full-graph message passing (kể cả test node tham gia propagate).
  Đây là thiết lập transductive — chuẩn cho GNN inductive learning cần
  dùng NeighborLoader. Test accuracy cao phản ánh phần nào sự kiện này.
  Xem metric 6-way Macro F1 để đánh giá thực chất hơn.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

# ===================== CONFIG (env-overridable for multi-seed sweeps) ==========
SEED = int(os.environ.get("GNN_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Architecture / training hyperparameters
HIDDEN_CHANNELS = int(os.environ.get("GNN_HIDDEN", "96"))
DROPOUT         = float(os.environ.get("GNN_DROPOUT", "0.4"))
GRL_ALPHA       = float(os.environ.get("GNN_GRL_ALPHA", "2.0"))
EDGE_DROPOUT    = float(os.environ.get("GNN_EDGE_DROPOUT", "0.2"))  # OOD regularization
LR              = float(os.environ.get("GNN_LR", "0.005"))
WEIGHT_DECAY    = float(os.environ.get("GNN_WD", "5e-4"))
W_SUB_ADV       = float(os.environ.get("GNN_W_ADV", "0.5"))   # adversarial confounder weight
# Suffix to keep multi-seed checkpoints/metrics from clobbering each other
RUN_TAG         = os.environ.get("GNN_RUN_TAG", "")
# Skip expensive counterfactual + causal-path attribution during HP sweeps
SKIP_EXPLAIN    = os.environ.get("GNN_SKIP_EXPLAIN", "0") == "1"
# FastRP is a TRANSDUCTIVE graph embedding computed on the full graph (incl. OOD
# nodes). For honest OOD/inductive evaluation it leaks cluster (=label) membership.
# DEFAULT = 0 (content-only: mpnet+CLIP+scalar) so reported metrics are leak-free.
# FastRP / PageRank / Louvain remain in the enriched CSVs for the BI dashboard only.
# Set GNN_USE_FASTRP=1 only for transductive in-distribution ablations.
USE_FASTRP      = os.environ.get("GNN_USE_FASTRP", "0") == "1"
# Backdoor adjustment: causal branch isolates the Subreddit confounder (structural
# do-calculus). DEFAULT on — this is what makes the causal head robust to confounder shift.
CAUSAL_CUT      = os.environ.get("GNN_CAUSAL_CUT", "1") == "1"
# Ablation: neutralize label-derived history features (target encoding P(fake|·)
# computed from train labels). Domain.fake_ratio_real / User.fake_rate are NOT cut
# by the Subreddit backdoor intervention, so the causal branch can still exploit
# them in transductive eval. Set =1 to replace with a constant 0.5 and measure how
# much of the OOD performance comes from these label-history channels.
NEUTRAL_DOMAIN  = os.environ.get("GNN_NEUTRAL_DOMAIN", "0") == "1"
NEUTRAL_USER    = os.environ.get("GNN_NEUTRAL_USER", "0") == "1"
# CLIP text-image consistency feature (03_clip_consistency.py → clip_cons.npy):
# cosine(CLIP-text(title), CLIP-image) — tín hiệu lệch ngữ nghĩa tiêu đề/ảnh (SAFE-style).
USE_CLIPCONS    = os.environ.get("GNN_USE_CLIPCONS", "0") == "1"
# AutoCut: thay phép cắt CỨNG Subreddit (biết trước) bằng cổng học được g_r∈[0,1]
# cho TỪNG loại quan hệ ở nhánh nhân quả. GRL adversarial đẩy gate→0 cho quan hệ
# mang tín hiệu môi trường; prior giữ-mở λ_open·Σ(1−g) giữ gate→1 cho quan hệ vô hại.
# → mô hình TỰ khám phá confounder thay vì được chỉ định.
AUTOCUT         = os.environ.get("GNN_AUTOCUT", "0") == "1"
AUTOCUT_OPEN    = float(os.environ.get("GNN_AUTOCUT_OPEN", "0.05"))
# Gates cần lr riêng cao hơn (scalar ít, gradient nhỏ) và KHÔNG early-stop:
# val (cùng phân phối confounded) luôn ưu ái shortcut → best-val sẽ chọn checkpoint
# trước khi gates kịp học. AutoCut dùng fixed-epoch (vẫn không oracle).
GATE_LR         = float(os.environ.get("GNN_GATE_LR", "0.05"))
AC_EPOCHS       = int(os.environ.get("GNN_AC_EPOCHS", "150"))
# GroupDRO trên causal 2-way loss: nhóm = (subreddit train × label), trọng số nhóm
# cập nhật nhân exp(η·L_g) → ép đều hiệu năng giữa các cộng đồng train.
GROUPDRO        = os.environ.get("GNN_GROUPDRO", "0") == "1"
GDRO_ETA        = float(os.environ.get("GNN_GDRO_ETA", "0.01"))

# ===================== PATHS =====================
# GNN_INPUT_DIR lets us point at an alternative processed dir (e.g. confounding-shift)
INPUT_DIR = os.environ.get("GNN_INPUT_DIR", os.path.join("data", "processed"))
OUTPUT_DIR = os.path.join("results")
MODEL_DIR  = os.path.join("models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# CLIP embedding dimension
CLIP_DIM = 512


# ===================== GRADIENT REVERSAL LAYER =====================

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)


# ===================== MODEL =====================

class CausalHeteroGNN(nn.Module):
    """
    Causal Heterogeneous GNN với Gradient Reversal Layer.

    Node types: Post, User, Subreddit, Domain, Image
    (Comment đã bị xoá — không có dữ liệu thực trong Fakeddit)

    Feature dims (default config: mpnet text encoder, FastRP excluded):
      Post:      768 (mpnet text) + 3 (scalar) = 771
                 (+64 FastRP only when GNN_USE_FASTRP=1 — leaky, ablation only)
      User:      4    (post_count, avg_score, avg_upvote_ratio, fake_rate)
      Subreddit: 3    (post_count, fake_ratio_real, avg_score)
      Domain:    3    (fake_ratio_real, avg_upvote_ratio, post_count)
      Image:     512 (CLIP ViT-B/32)
    """

    def __init__(self, metadata, hidden_channels=64, num_subreddits=100, dropout=0.3,
                 user_feat_dim=4, domain_feat_dim=3, post_feat_dim=451,
                 grl_alpha=2.0, edge_dropout=0.0, causal_cut=True, autocut=False):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout(p=dropout)
        self.edge_dropout = edge_dropout
        # Backdoor adjustment: the causal branch encodes on a graph where the
        # Subreddit confounder node is isolated (all edges touching Subreddit removed),
        # so the causal representation cannot use the spurious subreddit shortcut.
        self.causal_cut = causal_cut
        # AutoCut: learnable per-relation gates in the causal branch — the model
        # DISCOVERS which relation types to block instead of being told.
        # init +2.0 → sigmoid ≈ 0.88 (gần mở hẳn); adversarial GRL gradient đẩy
        # gate của quan hệ mang tín hiệu môi trường về 0.
        self.autocut = autocut
        self.edge_types = list(metadata[1])
        if autocut:
            self.gate_logits = nn.Parameter(
                torch.full((len(self.edge_types),), 2.0))
        # AutoCut v2 (structural search): cắt theo TÊN QUAN HỆ tùy chọn thay vì
        # node-type Subreddit cố định. cut_relations = set tên canonical
        # (vd {"posted_in","member_of"}); cạnh rev_* cùng quan hệ bị cắt theo.
        self.cut_relations = None

        # Feature projection — dimensions set at runtime based on available features
        # post_feat_dim = text_emb + fastrp(64) + scalar(3); varies with text encoder
        # (MiniLM 384 -> 451, mpnet 768 -> 835)
        self.proj = nn.ModuleDict({
            "Post":      nn.Linear(post_feat_dim,  hidden_channels),
            "User":      nn.Linear(user_feat_dim,  hidden_channels),
            "Subreddit": nn.Linear(3,              hidden_channels),
            "Domain":    nn.Linear(domain_feat_dim, hidden_channels),
            "Image":     nn.Linear(CLIP_DIM,       hidden_channels),
        })

        # HeteroConv layers (tự động dùng metadata từ HeteroData)
        def make_conv_dict():
            return {
                edge_type: SAGEConv(hidden_channels, hidden_channels)
                for edge_type in metadata[1]
            }

        self.conv1 = HeteroConv(make_conv_dict(), aggr="sum")
        self.conv2 = HeteroConv(make_conv_dict(), aggr="sum")

        # Causal / Spurious disentanglement heads
        def mlp(in_dim, out_dim, dropout_p=dropout):
            return nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(in_dim, out_dim),
            )

        self.causal_head   = mlp(hidden_channels, hidden_channels)
        self.spurious_head = mlp(hidden_channels, hidden_channels)

        # Confounder adversarial classifier (predict subreddit ID)
        self.confounder_clf = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels // 2, num_subreddits),
        )
        self.grl = GRL(alpha=grl_alpha)

        # 2-way classifiers
        self.baseline_clf_2way = mlp(hidden_channels, 2)
        self.causal_clf_2way   = mlp(hidden_channels, 2)

        # 6-way classifiers
        self.baseline_clf_6way = mlp(hidden_channels, 6)
        self.causal_clf_6way   = mlp(hidden_channels, 6)

    def _maybe_drop_edges(self, edge_index_dict):
        """Randomly drop edges during training → regularizes against over-reliance
        on the (confounded) graph structure, improving OOD generalization."""
        if not self.training or self.edge_dropout <= 0.0:
            return edge_index_dict
        from torch_geometric.utils import dropout_edge
        out = {}
        for k, ei in edge_index_dict.items():
            ei2, _ = dropout_edge(ei, p=self.edge_dropout, training=True)
            out[k] = ei2
        return out

    @staticmethod
    def _cut_confounder_edges(edge_index_dict):
        """Remove every edge incident on the Subreddit node (backdoor cut)."""
        return {k: v for k, v in edge_index_dict.items()
                if k[0] != "Subreddit" and k[2] != "Subreddit"}

    def _cut_relation_edges(self, edge_index_dict):
        """Cắt theo tên quan hệ (AutoCut v2). 'posted_in' cắt cả 'rev_posted_in'."""
        def canon(rel):
            return rel[4:] if rel.startswith("rev_") else rel
        return {k: v for k, v in edge_index_dict.items()
                if canon(k[1]) not in self.cut_relations}

    def encode(self, x_dict, edge_index_dict):
        edge_index_dict = self._maybe_drop_edges(edge_index_dict)
        h0 = {k: self.dropout(F.relu(self.proj[k](v)))
              for k, v in x_dict.items() if k in self.proj}
        h = self.conv1(h0, edge_index_dict)
        h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        # carry through any node type that received no edges this pass
        for k in h0:
            if k not in h:
                h[k] = h0[k]
        return h

    def _gated_layer(self, hetero_conv, h, edge_index_dict, gates):
        """Tái lập HeteroConv(aggr='sum') nhưng message của mỗi loại quan hệ
        được scale bởi gate g_r — dùng lại CHÍNH các SAGEConv bên trong
        hetero_conv (không thêm tham số conv mới). gates=1 ⇔ HeteroConv gốc."""
        out = {}
        for i, et in enumerate(self.edge_types):
            if et not in edge_index_dict:
                continue
            src, _, dst = et
            if src not in h or dst not in h:
                continue
            conv = hetero_conv.convs[et]  # PyG ModuleDict nhận tuple key trực tiếp
            m = conv((h[src], h[dst]), edge_index_dict[et])
            out[dst] = (out[dst] + gates[i] * m) if dst in out else gates[i] * m
        return out

    def encode_gated(self, x_dict, edge_index_dict):
        """Encoder nhánh nhân quả khi AutoCut bật: đồ thị ĐẦY ĐỦ nhưng mỗi loại
        quan hệ đi qua cổng học được. Quan hệ bị gate≈0 ⇔ bị cắt mềm."""
        edge_index_dict = self._maybe_drop_edges(edge_index_dict)
        gates = torch.sigmoid(self.gate_logits)
        h0 = {k: self.dropout(F.relu(self.proj[k](v)))
              for k, v in x_dict.items() if k in self.proj}
        h = self._gated_layer(self.conv1, h0, edge_index_dict, gates)
        h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        for k in h0:
            if k not in h:
                h[k] = h0[k]
        h2 = self._gated_layer(self.conv2, h, edge_index_dict, gates)
        for k in h0:
            if k not in h2:
                h2[k] = h[k]
        return h2

    def forward(self, x_dict, edge_index_dict):
        # Baseline branch: full graph (sees the Subreddit confounder).
        h_full = self.encode(x_dict, edge_index_dict)
        h_post = h_full["Post"]

        # Causal branch: graph with the Subreddit confounder isolated (backdoor cut),
        # hoặc AutoCut — cổng học được trên từng loại quan hệ (tự khám phá confounder).
        if self.cut_relations is not None:
            h_caus = self.encode(x_dict, self._cut_relation_edges(edge_index_dict))
            h_post_caus = h_caus["Post"]
        elif self.autocut:
            h_caus = self.encode_gated(x_dict, edge_index_dict)
            h_post_caus = h_caus["Post"]
        elif self.causal_cut:
            h_caus = self.encode(x_dict, self._cut_confounder_edges(edge_index_dict))
            h_post_caus = h_caus["Post"]
        else:
            h_post_caus = h_post

        h_c = self.causal_head(h_post_caus)
        h_s = self.spurious_head(h_post)

        pred_base_2   = self.baseline_clf_2way(h_post)
        pred_causal_2 = self.causal_clf_2way(h_c)
        pred_base_6   = self.baseline_clf_6way(h_post)
        pred_causal_6 = self.causal_clf_6way(h_c)

        sub_pred_spurious = self.confounder_clf(h_s)
        sub_pred_causal   = self.confounder_clf(self.grl(h_c))

        return (pred_base_2, pred_causal_2,
                pred_base_6, pred_causal_6,
                sub_pred_spurious, sub_pred_causal,
                h_c, h_s)


# ===================== BUILD HETERODATA =====================

def build_heterodata():
    print("Đọc enriched datasets...")
    posts_df      = pd.read_csv(os.path.join(INPUT_DIR, "posts_enriched.csv"))
    users_df      = pd.read_csv(os.path.join(INPUT_DIR, "users_enriched.csv"))
    subreddits_df = pd.read_csv(os.path.join(INPUT_DIR, "subreddits_enriched.csv"))
    domains_df    = pd.read_csv(os.path.join(INPUT_DIR, "domains_enriched.csv"))
    images_df     = pd.read_csv(os.path.join(INPUT_DIR, "images_enriched.csv"))

    post_embeddings = np.load(os.path.join(INPUT_DIR, "post_embeddings.npy"))
    post_fastrp     = np.load(os.path.join(INPUT_DIR, "post_fastrp.npy"))

    clip_path = os.path.join(INPUT_DIR, "image_embeddings.npy")
    if os.path.exists(clip_path):
        image_embeddings = np.load(clip_path)
        print(f"Đã tải CLIP embeddings: {image_embeddings.shape}")
    else:
        print("[CẢNH BÁO] image_embeddings.npy không tìm thấy, dùng random")
        image_embeddings = np.random.randn(len(images_df), CLIP_DIM).astype(np.float32)

    posted_by = pd.read_csv(os.path.join(INPUT_DIR, "posted_by.csv"))
    posted_in = pd.read_csv(os.path.join(INPUT_DIR, "posted_in.csv"))
    links_to  = pd.read_csv(os.path.join(INPUT_DIR, "links_to.csv"))
    has_image = pd.read_csv(os.path.join(INPUT_DIR, "has_image.csv"))
    member_of = pd.read_csv(os.path.join(INPUT_DIR, "member_of.csv"))

    # Map subreddit/domain về posts_df để OOD evaluation
    sub_name_map = (
        posted_in.merge(subreddits_df[["sub_id", "name"]], on="sub_id", how="left")
        .set_index("post_id")["name"].to_dict()
    )
    dom_name_map = (
        links_to.merge(domains_df[["domain_id", "url_domain"]], on="domain_id", how="left")
        .set_index("post_id")["url_domain"].to_dict()
    )
    posts_df["subreddit"] = posts_df["post_id"].apply(
        lambda x: sub_name_map.get(str(x), "unknown")
    )
    posts_df["domain"] = posts_df["post_id"].apply(
        lambda x: dom_name_map.get(str(x), "reddit.com")
    )

    # Index maps
    post_map   = {str(pid): i for i, pid in enumerate(posts_df["post_id"])}
    user_map   = {str(uid): i for i, uid in enumerate(users_df["user_id"])}
    sub_map    = {str(sid): i for i, sid in enumerate(subreddits_df["sub_id"])}
    domain_map = {str(did): i for i, did in enumerate(domains_df["domain_id"])}
    img_map    = {str(iid): i for i, iid in enumerate(images_df["img_id"])}

    data = HeteroData()

    # ---- Node Features ----

    # Post: text_emb(384) + fastrp(64) + [score, upvote_ratio, num_comments](3) = 451
    p_scalar = posts_df[["score", "upvote_ratio", "num_comments"]].values.astype(np.float32)
    p_scalar = _min_max_norm(p_scalar)
    if USE_FASTRP:
        p_feats = np.concatenate([post_embeddings, post_fastrp, p_scalar], axis=1)
        print(f"  Post features: text({post_embeddings.shape[1]}) + fastrp({post_fastrp.shape[1]}) "
              f"+ scalar({p_scalar.shape[1]}) = {p_feats.shape[1]} dims")
    else:
        p_feats = np.concatenate([post_embeddings, p_scalar], axis=1)
        print(f"  Post features: text({post_embeddings.shape[1]}) + scalar({p_scalar.shape[1]}) "
              f"= {p_feats.shape[1]} dims  [FastRP EXCLUDED — honest inductive/OOD]")
    if USE_CLIPCONS:
        cc_path = os.path.join(INPUT_DIR, "clip_cons.npy")
        cc = np.load(cc_path).astype(np.float32)
        p_feats = np.concatenate([p_feats, cc], axis=1)
        print(f"  + clip_cons(1) text-image consistency → {p_feats.shape[1]} dims")
    post_feat_dim = p_feats.shape[1]
    data["Post"].x = torch.tensor(p_feats, dtype=torch.float)
    data["Post"].y = torch.tensor(posts_df["label_2way"].values, dtype=torch.long)
    data["Post"].y_6way = torch.tensor(posts_df["label_6way"].values, dtype=torch.long)

    # Split masks
    data["Post"].train_mask = torch.tensor(
        (posts_df["split"] == "train").values, dtype=torch.bool)
    data["Post"].val_mask = torch.tensor(
        (posts_df["split"] == "val").values, dtype=torch.bool)
    data["Post"].test_mask = torch.tensor(
        (posts_df["split"] == "test").values, dtype=torch.bool)

    # OOD mask: test posts từ held-out subreddits (unseen trong training)
    posts_base_path = os.path.join(INPUT_DIR, "posts.csv")
    if os.path.exists(posts_base_path):
        posts_base = pd.read_csv(posts_base_path)
        if "is_ood" in posts_base.columns:
            ood_vals = posts_base["is_ood"].fillna(False).astype(bool).values
        else:
            ood_vals = np.zeros(len(posts_df), dtype=bool)
    else:
        ood_vals = np.zeros(len(posts_df), dtype=bool)
    data["Post"].ood_mask = torch.tensor(ood_vals, dtype=torch.bool)

    # Subreddit index per post (cho confounder classifier)
    post_to_sub_id = np.zeros(len(posts_df), dtype=np.int64)
    for _, r in posted_in.iterrows():
        pi = post_map.get(str(r["post_id"]))
        si = sub_map.get(str(r["sub_id"]))
        if pi is not None and si is not None:
            post_to_sub_id[pi] = si
    data["Post"].sub_id = torch.tensor(post_to_sub_id, dtype=torch.long)

    # User: [post_count, avg_score, avg_upvote_ratio, fake_rate] (4 proven features)
    # betweenness is available in enriched CSV for dashboard/BI use but NOT used as GNN input
    # because its extreme power-law distribution hurts model stability
    user_cols = ["post_count", "avg_score", "avg_upvote_ratio", "fake_rate"]
    u_df = users_df[user_cols].copy()
    if NEUTRAL_USER:
        u_df["fake_rate"] = 0.5
        print("  [ABLATION] User fake_rate neutralized to 0.5 (GNN_NEUTRAL_USER=1)")
    u_df["avg_upvote_ratio"] = u_df["avg_upvote_ratio"].fillna(u_df["avg_upvote_ratio"].median())
    u_df = u_df.fillna(0.0)
    u_feats = u_df.values.astype(np.float32)
    data["User"].x = torch.tensor(_min_max_norm(u_feats), dtype=torch.float)
    user_feat_dim = u_feats.shape[1]
    print(f"  User features: {user_cols} ({user_feat_dim} dims)")

    # Subreddit: [post_count, fake_ratio_real, avg_score] (3 features thực)
    s_feats = subreddits_df[["post_count", "fake_ratio_real", "avg_score"]].values.astype(np.float32)
    data["Subreddit"].x = torch.tensor(_min_max_norm(s_feats), dtype=torch.float)

    # Domain: [fake_ratio_real, avg_upvote_ratio, post_count] (3 features thực)
    # pagerank from GDS available in enriched CSV for dashboard but NOT as GNN input
    # (sparse signal — many domains have nearly identical GDS pagerank on small graph)
    domain_cols = ["fake_ratio_real", "avg_upvote_ratio", "post_count"]
    d_df = domains_df[domain_cols].copy()
    if NEUTRAL_DOMAIN:
        d_df["fake_ratio_real"] = 0.5
        print("  [ABLATION] Domain fake_ratio_real neutralized to 0.5 (GNN_NEUTRAL_DOMAIN=1)")
    d_feats = d_df.fillna(0.0).values.astype(np.float32)
    data["Domain"].x = torch.tensor(_min_max_norm(d_feats), dtype=torch.float)
    domain_feat_dim = d_feats.shape[1]
    print(f"  Domain features: {domain_cols} ({domain_feat_dim} dims)")

    # Image: CLIP embeddings (512-dim)
    data["Image"].x = torch.tensor(image_embeddings, dtype=torch.float)

    # ---- Edge Indices ----

    def build_edge(src_df, src_col, src_map, dst_col, dst_map):
        src = [src_map[str(x)] for x in src_df[src_col] if str(x) in src_map]
        dst = [dst_map[str(x)] for x in src_df[dst_col] if str(x) in dst_map]
        # Pair-wise filter: ambil hanya pasangan valid
        valid_src, valid_dst = [], []
        for s_val, d_val in zip(src_df[src_col], src_df[dst_col]):
            si = src_map.get(str(s_val))
            di = dst_map.get(str(d_val))
            if si is not None and di is not None:
                valid_src.append(si)
                valid_dst.append(di)
        return torch.tensor([valid_src, valid_dst], dtype=torch.long)

    data["Post",   "posted_by", "User"].edge_index      = build_edge(posted_by, "post_id", post_map, "user_id",   user_map)
    data["Post",   "posted_in", "Subreddit"].edge_index = build_edge(posted_in, "post_id", post_map, "sub_id",    sub_map)
    data["Post",   "links_to",  "Domain"].edge_index    = build_edge(links_to,  "post_id", post_map, "domain_id", domain_map)
    data["Post",   "has_image", "Image"].edge_index     = build_edge(has_image, "post_id", post_map, "img_id",    img_map)
    data["User",   "member_of", "Subreddit"].edge_index = build_edge(member_of, "user_id", user_map, "sub_id",    sub_map)

    # ToUndirected tạo reverse edges tự động
    data = T.ToUndirected()(data)

    return data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, user_feat_dim, domain_feat_dim, post_feat_dim


def _min_max_norm(arr):
    """Min-max normalize mỗi cột. Tránh chia cho 0."""
    mn  = arr.min(axis=0)
    mx  = arr.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return (arr - mn) / rng


# ===================== METRICS =====================

def compute_metrics(y_true, y_pred_logits):
    """Quick metrics dùng trong training loop."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred    = torch.argmax(y_pred_logits, dim=1).numpy()
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    acc = accuracy_score(y_true_np, y_pred)
    f1  = f1_score(y_true_np, y_pred, average="macro", zero_division=0)
    if y_pred_logits.shape[1] == 2:
        try:
            probs = F.softmax(y_pred_logits, dim=1)[:, 1].detach().numpy()
            auc   = roc_auc_score(y_true_np, probs)
        except Exception:
            auc = 0.5
    else:
        auc = 0.0
    return acc, f1, auc


def compute_full_metrics(y_true, y_pred_logits, label_names=None):
    """Comprehensive metrics: confusion matrix, per-class P/R/F1, AUC."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, classification_report, confusion_matrix,
    )
    y_pred    = torch.argmax(y_pred_logits, dim=1).numpy()
    y_true_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)

    results = {
        "accuracy":         float(accuracy_score(y_true_np, y_pred)),
        "macro_f1":         float(f1_score(y_true_np, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":      float(f1_score(y_true_np, y_pred, average="weighted", zero_division=0)),
        "macro_precision":  float(precision_score(y_true_np, y_pred, average="macro",    zero_division=0)),
        "macro_recall":     float(recall_score(y_true_np, y_pred, average="macro",       zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
    }

    report = classification_report(
        y_true_np, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    results["classification_report"] = report

    if y_pred_logits.shape[1] == 2:
        try:
            probs = F.softmax(y_pred_logits, dim=1)[:, 1].detach().numpy()
            results["auc_roc"] = float(roc_auc_score(y_true_np, probs))
        except Exception:
            results["auc_roc"] = 0.5
    else:
        results["auc_roc"] = None

    return results


def print_metrics_summary(name, m):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy:        {m['accuracy']:.4f}")
    print(f"  Macro F1:        {m['macro_f1']:.4f}")
    print(f"  Weighted F1:     {m['weighted_f1']:.4f}")
    print(f"  Macro Precision: {m['macro_precision']:.4f}")
    print(f"  Macro Recall:    {m['macro_recall']:.4f}")
    if m.get("auc_roc") is not None:
        print(f"  AUC-ROC:         {m['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    for row in m["confusion_matrix"]:
        print(f"    {row}")
    report = m.get("classification_report", {})
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':<22} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*60}")
    for key, vals in report.items():
        if isinstance(vals, dict) and "precision" in vals:
            print(f"  {key:<22} {vals['precision']:>8.4f} {vals['recall']:>8.4f} "
                  f"{vals['f1-score']:>8.4f} {vals['support']:>9.0f}")
    print(f"{'='*60}")


def build_masked_edge_dict(data, test_mask_bool):
    """
    Tạo edge_index_dict cho strict inductive evaluation:
    Xoá tất cả edges có nguồn (src) là test Post nodes.
    Test Posts vẫn nhận thông tin từ train neighbors (1-hop), nhưng
    không lan truyền ngược lại → ngăn test-to-train-to-test leakage.

    LƯU Ý: đây là định nghĩa "inductive một chiều" (test node vẫn NHẬN message).
    Các con số OOD báo cáo trong paper dùng định nghĩa CHẶT HƠN của
    07_evaluate.py / 06_baselines: mask cả hai chiều (content-only — test node
    bị cô lập hoàn toàn khỏi graph). Hai định nghĩa cho kết quả khác nhau;
    không trộn lẫn khi so sánh.
    """
    import copy
    test_np = test_mask_bool.cpu().numpy()
    masked  = {}
    for key, ei in data.edge_index_dict.items():
        src_type, _, dst_type = key
        if src_type == "Post":
            # Xoá edge nếu source là test post
            keep = ~torch.from_numpy(test_np[ei[0].cpu().numpy()])
            masked[key] = ei[:, keep]
        else:
            masked[key] = ei
    dc = copy.copy(data)
    dc.edge_index_dict = masked
    return dc


def compute_lfr(cf_results, model_key="causal"):
    """Tính Label Flip Rate (LFR) sau counterfactual interventions."""
    total = len(cf_results)
    if total == 0:
        return {"lfr_image": 0, "lfr_domain": 0, "lfr_subreddit": 0}

    def flipped(orig, interv):
        return (orig >= 0.5) != (interv >= 0.5)

    flip_img = sum(
        1 for r in cf_results
        if flipped(r["original"][model_key], r["cf_image"][model_key])
    )
    flip_dom = sum(
        1 for r in cf_results
        if flipped(r["original"][model_key], r["cf_domain"][model_key])
    )
    flip_sub = sum(
        1 for r in cf_results
        if flipped(r["original"][model_key], r["cf_subreddit"][model_key])
    )
    return {
        "lfr_image":     round(flip_img / total, 4),
        "lfr_domain":    round(flip_dom / total, 4),
        "lfr_subreddit": round(flip_sub / total, 4),
        "total_samples": total,
    }


# ===================== TRAINING =====================

def main():
    data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, \
        user_feat_dim, domain_feat_dim, post_feat_dim = build_heterodata()

    num_subreddits = len(subreddits_df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = CausalHeteroGNN(
        data.metadata(),
        hidden_channels=HIDDEN_CHANNELS,
        num_subreddits=num_subreddits,
        dropout=DROPOUT,
        user_feat_dim=user_feat_dim,
        domain_feat_dim=domain_feat_dim,
        post_feat_dim=post_feat_dim,
        grl_alpha=GRL_ALPHA,
        edge_dropout=EDGE_DROPOUT,
        causal_cut=CAUSAL_CUT,
        autocut=AUTOCUT,
    ).to(device)
    print(f"Config: hidden={HIDDEN_CHANNELS} dropout={DROPOUT} grl_alpha={GRL_ALPHA} "
          f"edge_dropout={EDGE_DROPOUT} lr={LR} wd={WEIGHT_DECAY} w_adv={W_SUB_ADV} "
          f"causal_cut={CAUSAL_CUT} autocut={AUTOCUT}(open={AUTOCUT_OPEN}) "
          f"clipcons={USE_CLIPCONS} groupdro={GROUPDRO} seed={SEED}")

    data = data.to(device)
    if AUTOCUT:
        gate_params  = [model.gate_logits]
        other_params = [p for n, p in model.named_parameters() if n != "gate_logits"]
        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
            {"params": gate_params,  "lr": GATE_LR, "weight_decay": 0.0},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)

    train_mask = data["Post"].train_mask
    val_mask   = data["Post"].val_mask
    test_mask  = data["Post"].test_mask

    y_train_2way = data["Post"].y[train_mask]
    y_train_6way = data["Post"].y_6way[train_mask]
    y_val_2way   = data["Post"].y[val_mask]
    y_test_2way  = data["Post"].y[test_mask]
    y_test_6way  = data["Post"].y_6way[test_mask]

    # Class weights cho 6-way (xử lý imbalance)
    counts_6way = torch.bincount(y_train_6way, minlength=6).float().to(device)
    weights_6way = (counts_6way.sum() / (6.0 * counts_6way + 1e-6))
    weights_6way = weights_6way / weights_6way.mean()
    print(f"6-way class counts (train): {counts_6way.cpu().long().tolist()}")
    print(f"6-way class weights:        {[round(w, 3) for w in weights_6way.cpu().tolist()]}")

    MAX_EPOCHS = AC_EPOCHS if AUTOCUT else 300
    PATIENCE   = 30
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    if AUTOCUT:
        print(f"[AutoCut] fixed-epoch training: {MAX_EPOCHS} epochs, "
              f"gate_lr={GATE_LR}, KHÔNG early-stop (val confounded ưu ái shortcut)")

    # GroupDRO: nhóm = (subreddit × label) trên train; trọng số nhóm thích nghi
    if GROUPDRO:
        g_train = (data["Post"].sub_id[train_mask] * 2 + y_train_2way)
        g_unique = torch.unique(g_train)
        g_masks = [(g_train == g) for g in g_unique]
        gdro_w = torch.ones(len(g_unique), device=device) / len(g_unique)
        print(f"GroupDRO: {len(g_unique)} nhóm (subreddit × label), eta={GDRO_ETA}")

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_f1_base": [], "val_f1_causal": [],
        "val_acc_base": [], "val_acc_causal": [],
    }

    print(f"\nBắt đầu train (max {MAX_EPOCHS} epochs, patience {PATIENCE})...")
    print(f"Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | "
          f"Test: {test_mask.sum().item()}")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        (pred_base_2, pred_causal_2,
         pred_base_6, pred_causal_6,
         sub_pred_spurious, sub_pred_causal,
         h_c, h_s) = model(data.x_dict, data.edge_index_dict)

        # --- Losses ---
        loss_base_2    = F.cross_entropy(pred_base_2[train_mask],   y_train_2way)
        loss_base_6    = F.cross_entropy(pred_base_6[train_mask],   y_train_6way,
                                          weight=weights_6way)
        if GROUPDRO:
            # GroupDRO: loss 2-way nhánh causal = tổng có trọng số theo nhóm;
            # trọng số nhóm tăng theo exp(eta * loss nhóm) → ưu tiên nhóm tệ nhất
            logits_tr = pred_causal_2[train_mask]
            g_losses = torch.stack([
                F.cross_entropy(logits_tr[m], y_train_2way[m]) for m in g_masks
            ])
            with torch.no_grad():
                gdro_w = gdro_w * torch.exp(GDRO_ETA * g_losses)
                gdro_w = gdro_w / gdro_w.sum()
            loss_causal_2 = (gdro_w * g_losses).sum()
        else:
            loss_causal_2 = F.cross_entropy(pred_causal_2[train_mask], y_train_2way)
        loss_causal_6  = F.cross_entropy(pred_causal_6[train_mask], y_train_6way,
                                          weight=weights_6way)

        sub_labels        = data["Post"].sub_id[train_mask]
        loss_sub_spurious = F.cross_entropy(sub_pred_spurious[train_mask], sub_labels)
        loss_sub_adv      = F.cross_entropy(sub_pred_causal[train_mask],   sub_labels)

        # Orthogonality loss: causal và spurious head không overlap
        loss_ortho = torch.mean(torch.abs(
            F.cosine_similarity(h_c[train_mask], h_s[train_mask])
        ))

        loss = (loss_base_2   + 0.5 * loss_base_6 +
                loss_causal_2 + 0.5 * loss_causal_6 +
                0.5 * loss_sub_spurious + W_SUB_ADV * loss_sub_adv +
                0.2 * loss_ortho)

        if AUTOCUT:
            # Prior giữ-mở: phạt việc đóng gate → chỉ những quan hệ mà GRL
            # adversarial "đòi" đóng mạnh hơn λ_open mới bị cắt
            gates_now = torch.sigmoid(model.gate_logits)
            loss = loss + AUTOCUT_OPEN * (1.0 - gates_now).sum()

        loss.backward()

        # Gradient clipping — ngăn explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_base_2, val_causal_2, val_base_6, val_causal_6, \
                val_sub_spu, val_sub_adv, vh_c, vh_s = \
                model(data.x_dict, data.edge_index_dict)
            val_acc_base,   val_f1_base,   _ = compute_metrics(
                y_val_2way.cpu(), val_base_2[val_mask].cpu())
            val_acc_causal, val_f1_causal, _ = compute_metrics(
                y_val_2way.cpu(), val_causal_2[val_mask].cpu())

            # Validation loss (causal 2-way + causal 6-way weighted)
            y_val_6way = data["Post"].y_6way[val_mask]
            val_loss = (
                F.cross_entropy(val_causal_2[val_mask], y_val_2way.to(device)).item() +
                0.5 * F.cross_entropy(val_causal_6[val_mask], y_val_6way.to(device),
                                      weight=weights_6way).item()
            )

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["val_loss"].append(float(val_loss))
        history["val_f1_base"].append(float(val_f1_base))
        history["val_f1_causal"].append(float(val_f1_causal))
        history["val_acc_base"].append(float(val_acc_base))
        history["val_acc_causal"].append(float(val_acc_causal))

        scheduler.step(val_loss)

        # Early stopping dựa theo validation loss (tránh plateau khi val F1 = 1.0).
        # AutoCut: LUÔN lưu checkpoint mới nhất + không early-stop — best-val sẽ
        # chọn model shortcut trước khi gates kịp học (val cùng phân phối confounded).
        if AUTOCUT:
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"causal_gnn{RUN_TAG}.pt"))
        elif val_loss < best_val_loss - 1e-4:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"causal_gnn{RUN_TAG}.pt"))
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val F1 Base: {val_f1_base:.4f} | "
                  f"Val F1 Causal: {val_f1_causal:.4f} | "
                  f"Patience: {epochs_no_improve}/{PATIENCE}")
        if AUTOCUT and epoch % 30 == 0:
            g = torch.sigmoid(model.gate_logits).detach().cpu()
            gd = sorted(zip(["__".join(e) for e in model.edge_types],
                            [float(x) for x in g]), key=lambda x: x[1])
            print("   gates:", " | ".join(f"{k.split('__')[1]}={v:.2f}" for k, v in gd))

        if not AUTOCUT and epochs_no_improve >= PATIENCE:
            print(f"\n*** Early stopping tại epoch {epoch} "
                  f"(val loss không cải thiện trong {PATIENCE} epochs) ***")
            break

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Đã lưu training_history.json ({len(history['epoch'])} epochs)")

    # Load best checkpoint
    print("\nLoad best model checkpoint...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"causal_gnn{RUN_TAG}.pt"),
                                     map_location=device))
    model.eval()

    # ===================== TEST EVALUATION =====================
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE TEST SET EVALUATION")
    print("  Transductive (full-graph) + OOD Subreddit Split")
    print("=" * 70)

    # OOD split dùng ood_mask đã build từ posts.csv is_ood column
    test_posts = posts_df[posts_df["split"] == "test"].copy().reset_index(drop=True)
    test_indices = np.where(test_mask.cpu().numpy())[0]

    ood_mask_full = data["Post"].ood_mask.cpu()
    seen_test_indices   = [int(i) for i in test_indices if not ood_mask_full[i]]
    unseen_test_indices = [int(i) for i in test_indices if ood_mask_full[i]]

    seen_mask   = torch.tensor(seen_test_indices,   dtype=torch.long)
    unseen_mask = torch.tensor(unseen_test_indices, dtype=torch.long)

    print(f"\nTest stats: total={len(test_indices)} | "
          f"seen={len(seen_test_indices)} | OOD-unseen={len(unseen_test_indices)}")

    if len(unseen_test_indices) > 0:
        ood_subs = test_posts[test_posts["subreddit"].isin(
            posts_df[ood_mask_full.numpy()]["subreddit"].unique()
        )]["subreddit"].unique().tolist()
        print(f"  OOD subreddits in test: {ood_subs}")

    # ── Transductive inference (full graph) ───────────────────────────────
    with torch.no_grad():
        (test_base_2, test_causal_2,
         test_base_6, test_causal_6, _, _, _, _) = model(data.x_dict, data.edge_index_dict)

    label_2way = ["Fake", "Real"]
    label_6way = ["True", "Satire", "Misleading", "Imposter", "FalseConn", "Manipulated"]

    m_base_2   = compute_full_metrics(y_test_2way.cpu(), test_base_2[test_mask].cpu(),   label_2way)
    m_causal_2 = compute_full_metrics(y_test_2way.cpu(), test_causal_2[test_mask].cpu(), label_2way)
    m_base_6   = compute_full_metrics(y_test_6way.cpu(), test_base_6[test_mask].cpu(),   label_6way)
    m_causal_6 = compute_full_metrics(y_test_6way.cpu(), test_causal_6[test_mask].cpu(), label_6way)

    print_metrics_summary("Baseline GNN — 2-Way [Transductive]", m_base_2)
    print_metrics_summary("Causal GNN   — 2-Way [Transductive]", m_causal_2)
    print_metrics_summary("Baseline GNN — 6-Way [Transductive]", m_base_6)
    print_metrics_summary("Causal GNN   — 6-Way [Transductive]", m_causal_6)

    # ── Strict Inductive inference (test-edges masked) ───────────────────
    print("\n--- Strict Inductive Eval (test post edges removed) ---")
    masked_data = build_masked_edge_dict(data, test_mask)
    with torch.no_grad():
        (si_base_2, si_causal_2,
         si_base_6, si_causal_6, _, _, _, _) = model(masked_data.x_dict,
                                                       masked_data.edge_index_dict)
    si_m_base_2   = compute_full_metrics(y_test_2way.cpu(), si_base_2[test_mask].cpu(),   label_2way)
    si_m_causal_2 = compute_full_metrics(y_test_2way.cpu(), si_causal_2[test_mask].cpu(), label_2way)
    si_m_base_6   = compute_full_metrics(y_test_6way.cpu(), si_base_6[test_mask].cpu(),   label_6way)
    si_m_causal_6 = compute_full_metrics(y_test_6way.cpu(), si_causal_6[test_mask].cpu(), label_6way)
    print_metrics_summary("Baseline GNN — 2-Way [Strict Inductive]", si_m_base_2)
    print_metrics_summary("Causal GNN   — 2-Way [Strict Inductive]", si_m_causal_2)
    print_metrics_summary("Baseline GNN — 6-Way [Strict Inductive]", si_m_base_6)
    print_metrics_summary("Causal GNN   — 6-Way [Strict Inductive]", si_m_causal_6)

    # ── OOD metrics (seen vs OOD-unseen subreddits) ─────────────────────
    ood_metrics = {}
    for tag, mask_t, label, logits_2 in [
        ("baseline_seen",   seen_mask,   "baseline", si_base_2),
        ("causal_seen",     seen_mask,   "causal",   si_causal_2),
        ("baseline_unseen", unseen_mask, "baseline", si_base_2),
        ("causal_unseen",   unseen_mask, "causal",   si_causal_2),
    ]:
        if len(mask_t) == 0:
            continue
        y_sub  = data["Post"].y[mask_t].cpu()
        logits = logits_2[mask_t].cpu()
        ood_metrics[tag] = compute_full_metrics(y_sub, logits, label_2way)

    # F1 drop
    f1_drop_base   = _f1_drop(ood_metrics, "baseline")
    f1_drop_causal = _f1_drop(ood_metrics, "causal")

    print(f"\n{'='*60}")
    print("  OOD GENERALIZATION SUMMARY (Strict Inductive)")
    print(f"{'='*60}")
    if "baseline_seen" in ood_metrics and "baseline_unseen" in ood_metrics:
        print(f"  Baseline — Seen F1: {ood_metrics['baseline_seen']['macro_f1']:.4f} | "
              f"OOD F1: {ood_metrics['baseline_unseen']['macro_f1']:.4f} | "
              f"Drop: {f1_drop_base:.2f}%")
        print(f"  Causal   — Seen F1: {ood_metrics['causal_seen']['macro_f1']:.4f} | "
              f"OOD F1: {ood_metrics['causal_unseen']['macro_f1']:.4f} | "
              f"Drop: {f1_drop_causal:.2f}%")
        improvement = f1_drop_base - f1_drop_causal
        print(f"  Causal GNN reduces F1 Drop by: {improvement:.2f}%")
    else:
        print("  [INFO] No OOD subreddits in test set — run with OOD split enabled")
    print(f"{'='*60}")

    # Metrics JSON
    metrics_res = _build_metrics_dict(
        m_base_2, m_causal_2, m_base_6, m_causal_6,
        si_m_base_2, si_m_causal_2, si_m_base_6, si_m_causal_6,
        ood_metrics, f1_drop_base, f1_drop_causal,
    )
    if AUTOCUT:
        gates_final = torch.sigmoid(model.gate_logits).detach().cpu()
        learned_gates = {"__".join(et): round(float(g), 4)
                         for et, g in zip(model.edge_types, gates_final)}
        metrics_res["learned_gates"] = learned_gates
        print("\nLearned relation gates (AutoCut):")
        for k, v in sorted(learned_gates.items(), key=lambda x: x[1]):
            marker = "  <-- CUT" if v < 0.2 else ""
            print(f"  {k:<42} {v:.3f}{marker}")
    with open(os.path.join(OUTPUT_DIR, f"metrics{RUN_TAG}.json"), "w") as f:
        json.dump(metrics_res, f, indent=4, default=str)
    print(f"\nĐã lưu metrics{RUN_TAG}.json")

    if SKIP_EXPLAIN:
        print("\n[SKIP_EXPLAIN=1] Bỏ qua counterfactual + causal path. Pipeline hoàn tất!")
        return

    # ===================== COUNTERFACTUAL ENGINE =====================
    print("\nTạo Counterfactual interventions...")
    cf_results = _run_counterfactuals(
        model, data, test_posts, test_indices,
        test_base_2, test_causal_2,
        subreddits_df, domains_df, sub_map, domain_map, device,
    )
    with open(os.path.join(OUTPUT_DIR, "counterfactuals.json"), "w") as f:
        json.dump(cf_results, f, indent=4)
    print("Đã lưu counterfactuals.json")

    # Label Flip Rate
    lfr_base   = compute_lfr(cf_results, "baseline")
    lfr_causal = compute_lfr(cf_results, "causal")
    print(f"\nLabel Flip Rate (LFR):")
    print(f"  Baseline — Image: {lfr_base['lfr_image']:.2%} | "
          f"Domain: {lfr_base['lfr_domain']:.2%} | "
          f"Subreddit: {lfr_base['lfr_subreddit']:.2%}")
    print(f"  Causal   — Image: {lfr_causal['lfr_image']:.2%} | "
          f"Domain: {lfr_causal['lfr_domain']:.2%} | "
          f"Subreddit: {lfr_causal['lfr_subreddit']:.2%}")

    # Ghi LFR vào metrics.json
    metrics_res["lfr"] = {"baseline": lfr_base, "causal": lfr_causal}
    with open(os.path.join(OUTPUT_DIR, f"metrics{RUN_TAG}.json"), "w") as f:
        json.dump(metrics_res, f, indent=4, default=str)

    # ===================== CAUSAL PATH ATTRIBUTION =====================
    print("\nTính Causal Path Attribution (gradient-based)...")
    causal_paths = _run_causal_paths(
        model, data, test_posts, test_indices,
        subreddits_df, domains_df, sub_map, domain_map, img_map,
    )
    with open(os.path.join(OUTPUT_DIR, "causal_paths.json"), "w") as f:
        json.dump(causal_paths, f, indent=4)
    print("Đã lưu causal_paths.json")

    print("\nPipeline hoàn tất!")


# ===================== HELPERS =====================

def _f1_drop(ood_metrics, model_key):
    seen_key   = f"{model_key}_seen"
    unseen_key = f"{model_key}_unseen"
    if seen_key in ood_metrics and unseen_key in ood_metrics:
        f1_s = ood_metrics[seen_key]["macro_f1"]
        f1_u = ood_metrics[unseen_key]["macro_f1"]
        return ((f1_s - f1_u) / f1_s * 100) if f1_s > 0 else 0.0
    return 0.0


def _build_metrics_dict(m_base_2, m_causal_2, m_base_6, m_causal_6,
                         si_m_base_2, si_m_causal_2, si_m_base_6, si_m_causal_6,
                         ood_metrics, f1_drop_base, f1_drop_causal):
    def ood_get(key, field):
        return ood_metrics.get(key, {}).get(field, 0)

    return {
        "evaluation_note": (
            "Two eval modes: (1) Transductive — full-graph message passing, "
            "accuracy inflated by train-to-test neighbor propagation. "
            "(2) Strict Inductive — test post edges removed, more honest. "
            "OOD F1 Drop (Strict Inductive, seen vs OOD subreddits) "
            "is the primary contribution metric."
        ),
        "baseline": {
            "overall_2way":          m_base_2,
            "overall_6way":          m_base_6,
            "strict_overall_2way":   si_m_base_2,
            "strict_overall_6way":   si_m_base_6,
            "seen_2way":             ood_metrics.get("baseline_seen",   {}),
            "unseen_2way":           ood_metrics.get("baseline_unseen", {}),
            "f1_drop_pct":           f1_drop_base,
            "overall": {
                "accuracy": si_m_base_2["accuracy"],
                "f1":       si_m_base_2["macro_f1"],
                "auc":      si_m_base_2.get("auc_roc", 0.5),
                "f1_6way":  si_m_base_6["macro_f1"],
            },
            "seen": {
                "accuracy": ood_get("baseline_seen", "accuracy"),
                "f1":       ood_get("baseline_seen", "macro_f1"),
                "auc":      ood_get("baseline_seen", "auc_roc"),
                "f1_6way":  0,
            },
            "unseen": {
                "accuracy": ood_get("baseline_unseen", "accuracy"),
                "f1":       ood_get("baseline_unseen", "macro_f1"),
                "auc":      ood_get("baseline_unseen", "auc_roc"),
                "f1_6way":  0,
            },
        },
        "causal": {
            "overall_2way":          m_causal_2,
            "overall_6way":          m_causal_6,
            "strict_overall_2way":   si_m_causal_2,
            "strict_overall_6way":   si_m_causal_6,
            "seen_2way":             ood_metrics.get("causal_seen",   {}),
            "unseen_2way":           ood_metrics.get("causal_unseen", {}),
            "f1_drop_pct":           f1_drop_causal,
            "overall": {
                "accuracy": si_m_causal_2["accuracy"],
                "f1":       si_m_causal_2["macro_f1"],
                "auc":      si_m_causal_2.get("auc_roc", 0.5),
                "f1_6way":  si_m_causal_6["macro_f1"],
            },
            "seen": {
                "accuracy": ood_get("causal_seen", "accuracy"),
                "f1":       ood_get("causal_seen", "macro_f1"),
                "auc":      ood_get("causal_seen", "auc_roc"),
                "f1_6way":  0,
            },
            "unseen": {
                "accuracy": ood_get("causal_unseen", "accuracy"),
                "f1":       ood_get("causal_unseen", "macro_f1"),
                "auc":      ood_get("causal_unseen", "auc_roc"),
                "f1_6way":  0,
            },
        },
    }


def _run_counterfactuals(model, data, test_posts, test_indices,
                          test_base_2, test_causal_2,
                          subreddits_df, domains_df, sub_map, domain_map, device):
    neutral_sub_id      = subreddits_df.loc[subreddits_df["fake_ratio_real"].idxmin(), "sub_id"]
    neutral_sub_idx     = sub_map[str(neutral_sub_id)]
    credible_domain_id  = domains_df.loc[domains_df["fake_ratio_real"].idxmin(), "domain_id"]
    credible_domain_idx = domain_map[str(credible_domain_id)]

    orig_edge_dict = {k: v.clone() for k, v in data.edge_index_dict.items()}
    cf_results     = []

    for local_idx, row_idx in enumerate(test_indices):
        post_row      = test_posts.iloc[local_idx]
        post_id       = str(post_row["post_id"])
        post_pyg_idx  = int(row_idx)

        p_orig_base   = F.softmax(test_base_2[post_pyg_idx].cpu(), dim=0)[1].item()
        p_orig_causal = F.softmax(test_causal_2[post_pyg_idx].cpu(), dim=0)[1].item()

        def cf_inference(cf_edge):
            with torch.no_grad():
                out_base, out_causal, _, _, _, _, _, _ = model(data.x_dict, cf_edge)
            return (F.softmax(out_base[post_pyg_idx].cpu(), dim=0)[1].item(),
                    F.softmax(out_causal[post_pyg_idx].cpu(), dim=0)[1].item())

        # CF1: do(image=None)
        cf_e = {k: v.clone() for k, v in orig_edge_dict.items()}
        img_key = ("Post", "has_image", "Image")
        rev_img = ("Image", "rev_has_image", "Post")
        if img_key in cf_e:
            mask = cf_e[img_key][0] != post_pyg_idx
            cf_e[img_key] = cf_e[img_key][:, mask]
        if rev_img in cf_e:
            mask = cf_e[rev_img][1] != post_pyg_idx
            cf_e[rev_img] = cf_e[rev_img][:, mask]
        p_img_base, p_img_causal = cf_inference(cf_e)

        # CF2: do(domain=credible)
        cf_e = {k: v.clone() for k, v in orig_edge_dict.items()}
        dom_key = ("Post", "links_to", "Domain")
        rev_dom = ("Domain", "rev_links_to", "Post")
        if dom_key in cf_e:
            e = cf_e[dom_key].clone()
            e[1, e[0] == post_pyg_idx] = credible_domain_idx
            cf_e[dom_key] = e
        if rev_dom in cf_e:
            e = cf_e[rev_dom].clone()
            e[0, e[1] == post_pyg_idx] = credible_domain_idx
            cf_e[rev_dom] = e
        p_dom_base, p_dom_causal = cf_inference(cf_e)

        # CF3: do(subreddit=neutral)
        cf_e = {k: v.clone() for k, v in orig_edge_dict.items()}
        sub_key = ("Post", "posted_in", "Subreddit")
        rev_sub = ("Subreddit", "rev_posted_in", "Post")
        if sub_key in cf_e:
            e = cf_e[sub_key].clone()
            e[1, e[0] == post_pyg_idx] = neutral_sub_idx
            cf_e[sub_key] = e
        if rev_sub in cf_e:
            e = cf_e[rev_sub].clone()
            e[0, e[1] == post_pyg_idx] = neutral_sub_idx
            cf_e[rev_sub] = e
        p_sub_base, p_sub_causal = cf_inference(cf_e)

        cf_results.append({
            "post_id":    post_id,
            "title":      post_row["title"],
            "subreddit":  post_row["subreddit"],
            "domain":     post_row["domain"],
            "label_true": int(post_row["label_2way"]),
            "original":   {"baseline": p_orig_base,   "causal": p_orig_causal},
            "cf_image":   {"baseline": p_img_base,    "causal": p_img_causal},
            "cf_domain":  {"baseline": p_dom_base,    "causal": p_dom_causal},
            "cf_subreddit": {"baseline": p_sub_base,  "causal": p_sub_causal},
        })

    return cf_results


def _run_causal_paths(model, data, test_posts, test_indices,
                       subreddits_df, domains_df, sub_map, domain_map, img_map):
    data["Subreddit"].x.requires_grad_(True)
    data["Domain"].x.requires_grad_(True)
    data["Image"].x.requires_grad_(True)

    model.train()
    _, pred_causal_2, _, _, _, _, _, _ = model(data.x_dict, data.edge_index_dict)

    causal_paths = {}
    for local_idx, row_idx in enumerate(test_indices):
        post_row     = test_posts.iloc[local_idx]
        post_id      = str(post_row["post_id"])
        post_pyg_idx = int(row_idx)

        prob_fake = F.softmax(pred_causal_2[post_pyg_idx], dim=0)[1]
        model.zero_grad()
        for t in ["Subreddit", "Domain", "Image"]:
            if data[t].x.grad is not None:
                data[t].x.grad.zero_()
        prob_fake.backward(retain_graph=True)

        sub_name = post_row["subreddit"]
        dom_name = post_row["domain"]
        si = sub_map.get(f"sub_{sub_name}", 0)
        di = domain_map.get(f"domain_{dom_name}", 0)
        ii = img_map.get(f"img_{post_id}", 0)

        grad_sub = data["Subreddit"].x.grad[si].abs().sum().item() \
            if data["Subreddit"].x.grad is not None else 0.0
        grad_dom = data["Domain"].x.grad[di].abs().sum().item() \
            if data["Domain"].x.grad is not None else 0.0
        grad_img = data["Image"].x.grad[ii].abs().sum().item() \
            if data["Image"].x.grad is not None else 0.0

        total = grad_sub + grad_dom + grad_img + 1e-8
        a_sub = max(0.0, min(1.0, grad_sub / total))
        a_dom = max(0.0, min(1.0, grad_dom / total))
        a_img = max(0.0, min(1.0, grad_img / total))

        # Dùng fake_ratio_real thay vì mock bias_strength
        sub_row = subreddits_df[subreddits_df["sub_id"] == f"sub_{sub_name}"]
        sub_fake = float(sub_row["fake_ratio_real"].values[0]) \
            if not sub_row.empty else 0.5

        dom_row = domains_df[domains_df["domain_id"] == f"domain_{dom_name}"]
        dom_fake = float(dom_row["fake_ratio_real"].values[0]) \
            if not dom_row.empty else 0.5

        if a_dom > a_sub and a_dom > a_img:
            explanation = (
                f"Flagged chủ yếu do nguồn domain. Domain '{dom_name}' có "
                f"fake_ratio={dom_fake:.2f} và chiếm {a_dom:.1%} influence."
            )
        elif a_img > a_sub and a_img > a_dom:
            explanation = (
                f"Flagged do yếu tố hình ảnh. Image chiếm {a_img:.1%} influence."
            )
        else:
            explanation = (
                f"Flagged do subreddit platform. r/{sub_name} có "
                f"fake_ratio={sub_fake:.2f} và chiếm {a_sub:.1%} influence."
            )

        causal_paths[post_id] = {
            "post_id":  post_id,
            "title":    post_row["title"],
            "subreddit": {"name": sub_name, "fake_ratio": sub_fake, "attribution": a_sub},
            "domain":    {"name": dom_name, "fake_ratio": dom_fake, "attribution": a_dom},
            "image":     {"attribution": a_img},
            "confidence": prob_fake.item(),
            "explanation": explanation,
        }

    return causal_paths


if __name__ == "__main__":
    main()
