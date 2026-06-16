"""
07_baselines_irm_eerm.py
========================
Baseline ĐỘC LẬP cho IRM (Invariant Risk Minimization) và EERM
(Explore-to-Extrapolate Risk Minimization) để so sánh CÔNG BẰNG với
CausalHeteroGNN trên cùng dữ liệu / cùng split.

THIẾT KẾ AN TOÀN — file này KHÔNG sửa pipeline gốc:
  - Chỉ *đọc* dữ liệu qua build_heterodata() của 05_train_gnn.py (read-only).
  - KHÔNG ghi gì lên Neo4j hay các file trong data/processed.
  - Tái dùng đúng backbone HeteroGraphSAGE (proj + 2 SAGEConv) để so sánh
    là khác biệt VỀ MẶT HỌC (invariance penalty) chứ không phải về kiến trúc.
  - KHÔNG dùng structural backdoor cut → đây chính là điểm tranh luận của bài:
    structural cut (hard) so với invariance penalty (soft) IRM/EERM.
  - Kết quả ghi riêng: results/baselines_irm_eerm{RUN_TAG}.json
    (cùng schema overall/seen/unseen/f1_drop_pct với metrics.json).

Hai phương pháp:
  • IRM  (Arjovsky et al. 2019): phạt độ lớn gradient của một classifier "dummy"
    (scale=1.0) trên TỪNG môi trường → buộc classifier tối ưu chung cho mọi env.
    Môi trường = phân hoạch train theo nhóm subreddit (confounder thay đổi giữa env).
  • EERM (Wu et al. 2022): sinh nhiều "môi trường ảo" trên đồ thị rồi tối thiểu
    rủi ro trung bình + phạt PHƯƠNG SAI rủi ro giữa các môi trường (risk extrapolation).
    Bản gốc huấn luyện edge-editer bằng REINFORCE; ở đây dùng biến thể thực dụng:
    sinh môi trường bằng nhiễu cạnh ngẫu nhiên (edge perturbation) — đã ghi chú rõ.

Biến môi trường (env vars) — giống các script khác:
  GNN_INPUT_DIR        : thư mục processed (mặc định data/processed)
  GNN_SEED             : seed (mặc định 42)
  GNN_RUN_TAG          : hậu tố file output cho sweep đa seed (vd "_s42")
  GNN_OOD_TRANSDUCTIVE : "1" → eval transductive (cho confounding-shift benchmark);
                         "0" (mặc định) → eval inductive content-only (standard OOD)
  GNN_USE_FASTRP       : kế thừa từ 05_train_gnn (mặc định 0 — leak-free)

Cách chạy:
  # Standard OOD split (inductive content-only)
  uv run python pipeline/07_baselines_irm_eerm.py

  # Confounding-shift benchmark (transductive, confounder hiển thị)
  $env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
  uv run python pipeline/07_baselines_irm_eerm.py
"""

import os
import json
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import dropout_edge

# Tái dùng loader + metrics từ 05_train_gnn (import động vì tên module bắt đầu bằng số)
train_gnn = importlib.import_module("05_train_gnn")
build_heterodata = train_gnn.build_heterodata
compute_full_metrics = train_gnn.compute_full_metrics
compute_metrics = train_gnn.compute_metrics

# ===================== CONFIG =====================
SEED    = int(os.environ.get("GNN_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

HIDDEN     = int(os.environ.get("GNN_HIDDEN", "96"))
DROPOUT    = float(os.environ.get("GNN_DROPOUT", "0.4"))
LR         = float(os.environ.get("GNN_LR", "0.005"))
WD         = float(os.environ.get("GNN_WD", "5e-4"))
MAX_EPOCHS = int(os.environ.get("GNN_MAX_EPOCHS", "300"))
PATIENCE   = int(os.environ.get("GNN_PATIENCE", "30"))
RUN_TAG    = os.environ.get("GNN_RUN_TAG", "")
TRANSDUCTIVE = os.environ.get("GNN_OOD_TRANSDUCTIVE", "0") == "1"

# IRM
N_ENV        = int(os.environ.get("IRM_N_ENV", "2"))       # số môi trường (phân hoạch subreddit)
IRM_LAMBDA   = float(os.environ.get("IRM_LAMBDA", "100.0"))# trọng số penalty sau anneal
IRM_ANNEAL   = int(os.environ.get("IRM_ANNEAL", "50"))     # epoch bắt đầu áp full penalty
# EERM
EERM_K       = int(os.environ.get("EERM_K", "3"))          # số môi trường ảo mỗi bước
EERM_BETA    = float(os.environ.get("EERM_BETA", "1.0"))   # trọng số phạt phương sai rủi ro
EERM_EDGE_P  = float(os.environ.get("EERM_EDGE_P", "0.3")) # mức nhiễu cạnh tối đa khi sinh env

OUTPUT_DIR = "results"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
CLIP_DIM = 512


# ===================== ENCODER (cùng backbone với CausalHeteroGNN) =====================

class HeteroEncoder(nn.Module):
    """proj + 2 lớp HeteroSAGE — KHÔNG có structural cut, KHÔNG có GRL/ortho.
    Trả về embedding của Post node. Đây là backbone dùng chung cho IRM & EERM."""

    def __init__(self, metadata, hidden, dropout, post_feat_dim,
                 user_feat_dim, domain_feat_dim):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.ModuleDict({
            "Post":      nn.Linear(post_feat_dim,   hidden),
            "User":      nn.Linear(user_feat_dim,   hidden),
            "Subreddit": nn.Linear(3,               hidden),
            "Domain":    nn.Linear(domain_feat_dim, hidden),
            "Image":     nn.Linear(CLIP_DIM,        hidden),
        })

        def make_conv():
            return HeteroConv(
                {et: SAGEConv(hidden, hidden) for et in metadata[1]}, aggr="sum")

        self.conv1 = make_conv()
        self.conv2 = make_conv()

    def forward(self, x_dict, edge_index_dict):
        h0 = {k: self.dropout(F.relu(self.proj[k](v)))
              for k, v in x_dict.items() if k in self.proj}
        h = self.conv1(h0, edge_index_dict)
        h = {k: self.dropout(F.relu(v)) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        for k in h0:
            if k not in h:
                h[k] = h0[k]
        return h["Post"]


class GNNClassifier(nn.Module):
    """Encoder + đầu phân loại 2-way (Fake/Real). Dùng cho cả IRM và EERM."""

    def __init__(self, metadata, hidden, dropout, post_feat_dim,
                 user_feat_dim, domain_feat_dim):
        super().__init__()
        self.encoder = HeteroEncoder(metadata, hidden, dropout, post_feat_dim,
                                     user_feat_dim, domain_feat_dim)
        self.clf = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_dict, edge_index_dict):
        h = self.encoder(x_dict, edge_index_dict)
        return self.clf(h)


# ===================== IRM =====================

def irm_penalty(logits, y):
    """IRMv1: ||∇_w CE(w·logits, y)||^2 tại w=1.0 (scale ảo)."""
    scale = torch.ones((), device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def make_environments(sub_id, train_mask, n_env):
    """Phân hoạch train Post thành n_env môi trường theo NHÓM subreddit.
    Mỗi môi trường chứa một tập subreddit rời nhau → mối tương quan
    subreddit→label khác nhau giữa các môi trường (điều kiện cần của IRM)."""
    train_idx = torch.where(train_mask)[0]
    subs = sub_id[train_idx].cpu().numpy()
    uniq = np.unique(subs)
    rng = np.random.RandomState(SEED)
    rng.shuffle(uniq)
    sub_to_env = {s: (i % n_env) for i, s in enumerate(uniq)}
    env_masks = []
    for e in range(n_env):
        sel = np.array([sub_to_env[s] == e for s in subs])
        env_masks.append(train_idx[torch.from_numpy(sel)])
    return env_masks


def train_irm(data, dims, train_mask, val_mask, device):
    metadata, post_dim, user_dim, dom_dim = dims
    model = GNNClassifier(metadata, HIDDEN, DROPOUT, post_dim, user_dim, dom_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    y = data["Post"].y
    env_masks = make_environments(data["Post"].sub_id, train_mask, N_ENV)
    print(f"[IRM] {N_ENV} môi trường, kích thước: {[len(m) for m in env_masks]}")

    best_val, no_improve = float("inf"), 0
    ckpt = os.path.join(MODEL_DIR, f"irm{RUN_TAG}.pt")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        nlls, pens = [], []
        for em in env_masks:
            if len(em) == 0:
                continue
            le = logits[em]; ye = y[em]
            nlls.append(F.cross_entropy(le, ye))
            pens.append(irm_penalty(le, ye))
        mean_nll = torch.stack(nlls).mean()
        mean_pen = torch.stack(pens).mean()
        pw = IRM_LAMBDA if epoch >= IRM_ANNEAL else 1.0
        loss = mean_nll + pw * mean_pen
        if pw > 1.0:
            loss = loss / pw  # giữ thang loss ổn định (chuẩn IRM)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vlog = model(data.x_dict, data.edge_index_dict)
            vloss = F.cross_entropy(vlog[val_mask], y[val_mask]).item()
            vacc, vf1, _ = compute_metrics(y[val_mask].cpu(), vlog[val_mask].cpu())
        sched.step(vloss)
        if vloss < best_val - 1e-4:
            best_val, no_improve = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  [IRM] ep{epoch:03d} nll={mean_nll.item():.3f} "
                  f"pen={mean_pen.item():.2e} vloss={vloss:.3f} vf1={vf1:.3f} "
                  f"pat={no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print(f"  [IRM] early stop @ ep{epoch}")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


# ===================== EERM =====================

def perturb_edges(edge_index_dict, p):
    """Sinh một 'môi trường ảo' bằng cách bỏ ngẫu nhiên cạnh (edge perturbation).
    Đây là biến thể thực dụng của bộ sinh môi trường EERM (bản gốc dùng edge-editer
    huấn luyện bằng REINFORCE); ghi chú rõ trong docstring module."""
    out = {}
    for k, ei in edge_index_dict.items():
        ei2, _ = dropout_edge(ei, p=p, training=True)
        out[k] = ei2
    return out


def train_eerm(data, dims, train_mask, val_mask, device):
    metadata, post_dim, user_dim, dom_dim = dims
    model = GNNClassifier(metadata, HIDDEN, DROPOUT, post_dim, user_dim, dom_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
    y = data["Post"].y
    print(f"[EERM] K={EERM_K} môi trường ảo/bước, beta={EERM_BETA}, edge_p<= {EERM_EDGE_P}")

    best_val, no_improve = float("inf"), 0
    ckpt = os.path.join(MODEL_DIR, f"eerm{RUN_TAG}.pt")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        risks = []
        for k in range(EERM_K):
            # Mỗi env có mức nhiễu khác nhau → đa dạng phân bố (explore)
            p = EERM_EDGE_P * (k + 1) / EERM_K
            ed = perturb_edges(data.edge_index_dict, p)
            logits = model(data.x_dict, ed)
            risks.append(F.cross_entropy(logits[train_mask], y[train_mask]))
        risks = torch.stack(risks)
        mean_risk = risks.mean()
        var_risk = risks.var(unbiased=False)
        loss = mean_risk + EERM_BETA * var_risk  # extrapolate: phạt phương sai rủi ro
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            vlog = model(data.x_dict, data.edge_index_dict)
            vloss = F.cross_entropy(vlog[val_mask], y[val_mask]).item()
            vacc, vf1, _ = compute_metrics(y[val_mask].cpu(), vlog[val_mask].cpu())
        sched.step(vloss)
        if vloss < best_val - 1e-4:
            best_val, no_improve = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_improve += 1
        if epoch % 20 == 0:
            print(f"  [EERM] ep{epoch:03d} mean_risk={mean_risk.item():.3f} "
                  f"var={var_risk.item():.2e} vloss={vloss:.3f} vf1={vf1:.3f} "
                  f"pat={no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print(f"  [EERM] early stop @ ep{epoch}")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


# ===================== EVALUATION =====================

def mask_post_edges(edge_index_dict, post_ids):
    """Xoá mọi cạnh chạm vào các test Post node (inductive content-only)."""
    post_ids = set(int(i) for i in post_ids)
    out = {}
    for et, ei in edge_index_dict.items():
        s, _, d = et
        if s == "Post":
            keep = torch.tensor([int(x) not in post_ids for x in ei[0]], dtype=torch.bool)
            out[et] = ei[:, keep]
        elif d == "Post":
            keep = torch.tensor([int(x) not in post_ids for x in ei[1]], dtype=torch.bool)
            out[et] = ei[:, keep]
        else:
            out[et] = ei
    return out


def evaluate(model, data, posts_df, device):
    """Trả về dict {overall, seen, unseen, f1_drop_pct} giống schema metrics.json."""
    model.eval()
    test_mask = data["Post"].test_mask
    y = data["Post"].y
    test_indices = torch.where(test_mask)[0].cpu().numpy()

    # seen vs OOD-unseen — ưu tiên cờ is_ood (cũng dùng cho confounding-shift)
    test_posts = posts_df[posts_df["split"] == "test"].copy()
    if "is_ood" in test_posts.columns and test_posts["is_ood"].fillna(False).astype(bool).any():
        is_seen = ~test_posts["is_ood"].fillna(False).astype(bool).values
    else:
        train_subs = set(posts_df[posts_df["split"] == "train"]["subreddit"])
        is_seen = test_posts["subreddit"].isin(train_subs).values

    seen_idx   = [int(i) for i, s in zip(test_indices, is_seen) if s]
    unseen_idx = [int(i) for i, s in zip(test_indices, is_seen) if not s]

    # Chọn chế độ inference (khớp với 07_evaluate.py)
    if TRANSDUCTIVE:
        with torch.no_grad():
            logits = model(data.x_dict, data.edge_index_dict)
        print("  [eval] TRANSDUCTIVE (confounder visible) — confounding-shift mode")
    else:
        ed = mask_post_edges(data.edge_index_dict, test_indices)
        with torch.no_grad():
            logits = model(data.x_dict, ed)
        print("  [eval] INDUCTIVE content-only (leak-free) — standard OOD mode")

    labels2 = ["Fake", "Real"]
    overall = compute_full_metrics(y[test_mask].cpu(), logits[test_mask].cpu(), labels2)

    def sub(idx):
        if not idx:
            return {}
        m = torch.tensor(idx, dtype=torch.long)
        return compute_full_metrics(y[m].cpu(), logits[m].cpu(), labels2)

    seen_m   = sub(seen_idx)
    unseen_m = sub(unseen_idx)
    f1_drop = 0.0
    if seen_m and unseen_m and seen_m["macro_f1"] > 0:
        f1_drop = (seen_m["macro_f1"] - unseen_m["macro_f1"]) / seen_m["macro_f1"] * 100

    return {
        "overall": {"accuracy": overall["accuracy"], "f1": overall["macro_f1"],
                    "auc": overall.get("auc_roc", 0.5)},
        "overall_2way": overall,
        "seen": {"accuracy": seen_m.get("accuracy", 0), "f1": seen_m.get("macro_f1", 0),
                 "auc": seen_m.get("auc_roc", 0)},
        "unseen": {"accuracy": unseen_m.get("accuracy", 0), "f1": unseen_m.get("macro_f1", 0),
                   "auc": unseen_m.get("auc_roc", 0)},
        "seen_2way": seen_m,
        "unseen_2way": unseen_m,
        "f1_drop_pct": f1_drop,
    }


# ===================== MAIN =====================

def main():
    data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, \
        user_feat_dim, domain_feat_dim, post_feat_dim = build_heterodata()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device} | seed={SEED} | transductive_eval={TRANSDUCTIVE}")
    data = data.to(device)
    dims = (data.metadata(), post_feat_dim, user_feat_dim, domain_feat_dim)
    train_mask = data["Post"].train_mask
    val_mask   = data["Post"].val_mask

    print("\n" + "=" * 60 + "\n  TRAIN IRM\n" + "=" * 60)
    irm_model = train_irm(data, dims, train_mask, val_mask, device)
    irm_metrics = evaluate(irm_model, data, posts_df, device)

    print("\n" + "=" * 60 + "\n  TRAIN EERM\n" + "=" * 60)
    eerm_model = train_eerm(data, dims, train_mask, val_mask, device)
    eerm_metrics = evaluate(eerm_model, data, posts_df, device)

    out = {
        "_config": {
            "seed": SEED, "transductive_eval": TRANSDUCTIVE,
            "input_dir": os.environ.get("GNN_INPUT_DIR", "data/processed"),
            "irm": {"n_env": N_ENV, "lambda": IRM_LAMBDA, "anneal": IRM_ANNEAL},
            "eerm": {"K": EERM_K, "beta": EERM_BETA, "edge_p": EERM_EDGE_P},
            "note": ("Standalone IRM/EERM baselines, same HeteroGraphSAGE backbone, "
                     "NO structural cut. EERM uses random edge-perturbation env "
                     "generation (practical variant of the REINFORCE edge-editer)."),
        },
        "irm": irm_metrics,
        "eerm": eerm_metrics,
    }
    out_path = os.path.join(OUTPUT_DIR, f"baselines_irm_eerm{RUN_TAG}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4, default=str)
    print(f"\nĐã lưu {out_path}")

    # ---- In bảng so sánh (kèm CausalHeteroGNN nếu có metrics{RUN_TAG}.json) ----
    print("\n" + "=" * 72)
    print("  SO SÁNH OOD (unseen) — IRM / EERM" +
          (" / Causal+Baseline" if os.path.exists(
              os.path.join(OUTPUT_DIR, f"metrics{RUN_TAG}.json")) else ""))
    print("=" * 72)
    header = f"  {'Model':<16}{'OOD Acc':>10}{'OOD F1':>10}{'OOD AUC':>10}{'F1 Drop%':>10}"
    print(header); print("  " + "-" * 54)

    def row(name, m):
        u = m["unseen"]
        print(f"  {name:<16}{u['accuracy']:>10.4f}{u['f1']:>10.4f}"
              f"{u['auc']:>10.4f}{m['f1_drop_pct']:>10.2f}")

    causal_path = os.path.join(OUTPUT_DIR, f"metrics{RUN_TAG}.json")
    if os.path.exists(causal_path):
        cm = json.load(open(causal_path))
        if "baseline" in cm:
            row("Baseline GNN", cm["baseline"])
        if "causal" in cm:
            row("CausalHeteroGNN", cm["causal"])
    row("IRM", irm_metrics)
    row("EERM", eerm_metrics)
    print("=" * 72)
    print("\nGhi chú: chạy đa seed với GNN_RUN_TAG khác nhau, rồi gộp:")
    print("  python aggregate_seeds.py results/baselines_irm_eerm_s42.json ...")


if __name__ == "__main__":
    main()
