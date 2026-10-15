"""
02_train_upfd.py
================
Phân loại tin giả dạng PHÂN LOẠI ĐỒ THỊ trên UPFD: mỗi tin là một cây lan truyền
(node gốc = tin, các node khác = user đã retweet), nhãn gắn cho cả đồ thị (1 = Fake, 0 = Real).

Mô hình (--model):
  root_mlp   MLP chỉ trên feature node gốc (tin) — trả lời "đồ thị có giúp gì không?"
  stats_mlp  MLP trên vài thống kê cấu trúc (số node, số cạnh, bậc gốc, độ sâu) — lối tắt cấu trúc
  gcn|sage|gat
             GNN 2 lớp trên cây lan truyền -> global max pool, nối với Linear(feature gốc)
             (thiết kế UPFD: "news + user engagement"), rồi MLP -> 2 lớp
  causal     = sage + bộ phân biệt môi trường qua GRL (ý tưởng của CausalHeteroGNN, nhưng
             confounder là MÔI TRƯỜNG THỜI GIAN thay vì node Subreddit): ép biểu diễn đồ thị
             không mang thông tin "tin này thuộc giai đoạn nào". α theo lịch DANN mặc định
             (chẩn đoán MediaEval cho thấy α = 2 hằng số làm sụp mô hình).

Chia dữ liệu (--split):
  original   chia gốc của tác giả UPFD (20/10/70, ngẫu nhiên)
  temporal   theo thời điểm retweet đầu tiên: 60% cũ nhất train, 10% val, 30% mới nhất test (OOD)
Môi trường cho causal: --n_env khoảng thời gian, cắt theo phân vị trên tập train.

Ví dụ:
  uv run python propagation/02_train_upfd.py --dataset politifact --feature bert --model sage
  uv run python propagation/02_train_upfd.py --dataset gossipcop --split temporal --model causal
Output: results/upfd/runs/{dataset}_{feature}_{split}_{model}{tag}_s{seed}.json
"""

import os
import json
import copy
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
from torch_geometric.utils import degree, dropout_edge

ROOT = os.path.join("data", "upfd")
RUN_DIR = os.path.join("results", "upfd", "runs")
os.makedirs(RUN_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["politifact", "gossipcop"], default="politifact")
    p.add_argument("--feature", choices=["bert", "spacy", "profile", "content"], default="bert")
    p.add_argument("--model", choices=["root_mlp", "stats_mlp", "gcn", "sage", "gat", "irm", "eerm", "causal"],
                   default="sage")
    p.add_argument("--split", choices=["original", "temporal"], default="original")
    p.add_argument("--n_env", type=int, default=4)
    p.add_argument("--grl_alpha", type=float, default=1.0)
    p.add_argument("--grl_schedule", choices=["const", "dann"], default="dann")
    p.add_argument("--w_adv", type=float, default=0.5)
    p.add_argument("--adv_conditional", action="store_true",
                   help="bộ phân biệt môi trường nhận thêm nhãn y -> chỉ xoá thông tin thời gian VƯỢT QUÁ nhãn")
    p.add_argument("--test_dataset", choices=["politifact", "gossipcop"], default=None,
                   help="cross-dataset: train/val từ --dataset, test = toàn bộ đồ thị của bộ này")
    # IRM / EERM: cùng siêu tham số với pipeline/07_baselines_irm_eerm.py (anneal co theo 60 epoch)
    p.add_argument("--irm_lambda", type=float, default=100.0)
    p.add_argument("--irm_anneal", type=int, default=10)
    p.add_argument("--eerm_k", type=int, default=3)
    p.add_argument("--eerm_beta", type=float, default=1.0)
    p.add_argument("--eerm_p", type=float, default=0.3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--tag", default="")
    return p.parse_args()


# ===================== DATA =====================

def read_dataset(name, feature):
    meta = pd.read_csv(os.path.join(ROOT, name, "graph_meta.csv"))
    graphs = []
    for s in ["train", "val", "test"]:
        ds = UPFD(ROOT, name, feature, s)
        idx = np.load(os.path.join(ROOT, name, "raw", f"{s}_idx.npy"))
        for k in range(len(ds)):
            d = ds[k]
            d.graph_id = int(idx[k])
            d.orig_split = s
            graphs.append(d)
    t = meta.set_index("graph_id")["first_retweet"]
    return graphs, t.fillna(t.median()), meta


def load_graphs(args):
    """Nạp 3 split gốc, gắn graph_id gốc + metadata thời gian, rồi chia lại nếu cần."""
    graphs, t, meta = read_dataset(args.dataset, args.feature)
    ext = None
    if args.test_dataset and args.test_dataset != args.dataset:
        ext, t_ext, _ = read_dataset(args.test_dataset, args.feature)

    rng = np.random.RandomState(0)

    def seen_split(ids):
        """Chia ngẫu nhiên (cố định) 75/10/15 -> train / val / seen-test (in-distribution)."""
        ids = rng.permutation(np.array(ids))
        a, b = int(0.75 * len(ids)), int(0.85 * len(ids))
        return {**{g: "train" for g in ids[:a]}, **{g: "val" for g in ids[a:b]},
                **{g: "seen" for g in ids[b:]}}

    if ext is not None:
        # cross-dataset: --dataset chia train/val/seen; test (OOD) = toàn bộ bộ kia
        split = seen_split([d.graph_id for d in graphs])
    elif args.split == "original":
        split = {d.graph_id: d.orig_split for d in graphs}
    else:
        # temporal: 70% tin cũ nhất chia train/val/seen; 30% mới nhất = test (OOD)
        order = t.loc[[d.graph_id for d in graphs]].sort_values().index.tolist()
        cut = int(0.7 * len(order))
        split = {**seen_split(order[:cut]), **{g: "test" for g in order[cut:]}}

    # môi trường = khoảng thời gian, cắt theo phân vị của tập train
    train_t = t.loc[[g for g, s in split.items() if s == "train"]].values
    cuts = np.quantile(train_t, np.linspace(0, 1, args.n_env + 1)[1:-1])
    # nhóm kích thước (dưới/trên median số node của train) -> worst-group theo (kích thước × nhãn)
    med = np.median([d.num_nodes for d in graphs if split[d.graph_id] == "train"])
    out = {"train": [], "val": [], "seen": [], "test": []}
    for d in graphs:
        d.env = torch.tensor([int(np.searchsorted(cuts, t.loc[d.graph_id], side="right"))])
        d.size_grp = torch.tensor([int(d.num_nodes > med)])
        d.stats = graph_stats(d)
        out[split[d.graph_id]].append(d)
    if ext is not None:
        for d in ext:
            d.env = torch.tensor([int(np.searchsorted(cuts, t_ext.loc[d.graph_id], side="right"))])
            d.size_grp = torch.tensor([int(d.num_nodes > med)])
            d.stats = graph_stats(d)
            out["test"].append(d)
    return out, meta


def graph_stats(d):
    n = d.num_nodes
    e = d.edge_index.shape[1]
    deg = degree(d.edge_index[0], n) + degree(d.edge_index[1], n)
    return torch.tensor([[np.log1p(n), np.log1p(e), float(deg[0]) / max(n - 1, 1),
                          float(deg.max()) / max(n - 1, 1)]], dtype=torch.float)


# ===================== MODELS =====================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.alpha, None


def mlp(i, h, o, p):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


class RootMLP(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.net = mlp(d, h, 2, p)

    def forward(self, b):
        return self.net(b.x[b.ptr[:-1]]), None


class StatsMLP(nn.Module):
    def __init__(self, h, p):
        super().__init__()
        self.net = mlp(4, h, 2, p)

    def forward(self, b):
        return self.net(b.stats), None


class PropGNN(nn.Module):
    """UPFD-style: GNN trên cây lan truyền + max pool, nối với Linear(feature tin gốc)."""

    def __init__(self, d, h, p, conv="sage", n_env=0, conditional=False):
        super().__init__()
        Conv = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}[conv]
        self.c1, self.c2 = Conv(d, h), Conv(h, h)
        self.root = nn.Linear(d, h)
        self.body = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(p))
        self.clf = nn.Linear(h, 2)
        self.conditional = conditional
        self.env_clf = mlp(h + (2 if conditional else 0), h // 2, n_env, p) if n_env else None
        self.p = p
        self.alpha = 0.0

    def forward(self, b):
        x = F.relu(self.c1(b.x, b.edge_index))
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.c2(x, b.edge_index))
        g = global_max_pool(x, b.batch)
        r = F.relu(self.root(b.x[b.ptr[:-1]]))
        z = self.body(torch.cat([g, r], dim=1))
        env = None
        if self.env_clf is not None:
            zr = GradReverse.apply(z, self.alpha)
            if self.conditional:
                zr = torch.cat([zr, F.one_hot(b.y, 2).float()], dim=1)
            env = self.env_clf(zr)
        return self.clf(z), env


def build_model(args, d):
    if args.model == "root_mlp":
        return RootMLP(d, args.hidden, args.dropout)
    if args.model == "stats_mlp":
        return StatsMLP(args.hidden, args.dropout)
    if args.model == "causal":
        return PropGNN(d, args.hidden, args.dropout, "sage", n_env=args.n_env,
                       conditional=args.adv_conditional)
    conv = "sage" if args.model in ("irm", "eerm") else args.model
    return PropGNN(d, args.hidden, args.dropout, conv)


def irm_penalty(logits, y):
    scale = torch.ones((), requires_grad=True)
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


# ===================== TRAIN / EVAL =====================

@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, envs, sgs = [], [], [], []
    for b in loader:
        logits, _ = model(b)
        ys.append(b.y); envs.append(b.env); sgs.append(b.size_grp)
        ps.append(F.softmax(logits, 1)[:, 1])
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(envs).numpy(),
            torch.cat(sgs).numpy())


def metrics(y, p, env, sg):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    pred = (p >= 0.5).astype(int)
    m = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
         "auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
         "recall_fake": float((pred[y == 1] == 1).mean()) if (y == 1).any() else None,
         "recall_real": float((pred[y == 0] == 0).mean()) if (y == 0).any() else None,
         "pred_share_fake": float(pred.mean())}
    groups = {f"size{s}_y{c}": float((pred[(sg == s) & (y == c)] == c).mean())
              for s in (0, 1) for c in (0, 1) if ((sg == s) & (y == c)).sum() >= 5}
    m["acc_by_size_label"] = groups
    m["worst_group_acc"] = min(groups.values()) if groups else None
    per_env = {int(e): float((pred[env == e] == y[env == e]).mean()) for e in np.unique(env)}
    m["acc_by_env"] = per_env
    m["worst_env_acc"] = min(per_env.values())
    m["collapsed"] = bool(m["macro_f1"] < 0.5 or max(m["pred_share_fake"], 1 - m["pred_share_fake"]) > 0.9)
    return m


def run_seed(args, data, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = data["train"][0].num_features
    model = build_model(args, d)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    tl = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True)
    vl = DataLoader(data["val"], batch_size=512)
    te = DataLoader(data["test"], batch_size=512)
    se = DataLoader(data["seen"], batch_size=512) if data["seen"] else None
    best, best_state, best_ep, hist = -1.0, None, 0, []
    steps = args.epochs * len(tl)
    step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, tot_adv = 0.0, 0.0
        for b in tl:
            step += 1
            if args.model == "causal":
                prog = step / steps
                model.alpha = (args.grl_alpha if args.grl_schedule == "const"
                               else args.grl_alpha * (2 / (1 + np.exp(-10 * prog)) - 1))
            opt.zero_grad()
            logits, env_logits = model(b)
            loss = F.cross_entropy(logits, b.y)
            if args.model == "irm":
                pen = sum(irm_penalty(logits[b.env == e], b.y[b.env == e])
                          for e in b.env.unique() if (b.env == e).sum() > 1)
                w = args.irm_lambda if ep >= args.irm_anneal else 1.0
                loss = (loss + w * pen) / (w if w > 1 else 1.0)
            elif args.model == "eerm":
                risks = []
                for k in range(args.eerm_k):
                    bk = b.clone()
                    bk.edge_index, _ = dropout_edge(b.edge_index, p=args.eerm_p * (k + 1) / args.eerm_k)
                    risks.append(F.cross_entropy(model(bk)[0], b.y))
                risks = torch.stack(risks)
                loss = risks.mean() + args.eerm_beta * risks.var(unbiased=False)
            if env_logits is not None:
                l_adv = F.cross_entropy(env_logits, b.env)
                loss = loss + args.w_adv * l_adv
                tot_adv += l_adv.item() * b.num_graphs
            loss.backward()
            opt.step()
            tot += loss.item() * b.num_graphs
        y, p, e, sg = predict(model, vl)
        vm = metrics(y, p, e, sg)
        hist.append({"epoch": ep, "loss": tot / len(data["train"]),
                     "L_adv": tot_adv / len(data["train"]) if args.model == "causal" else None,
                     "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"]})
        if vm["macro_f1"] > best:
            best, best_ep, best_state = vm["macro_f1"], ep, copy.deepcopy(model.state_dict())
    last = metrics(*predict(model, te))
    model.load_state_dict(best_state)
    return {"config": vars(args), "seed": seed, "best_epoch": best_ep,
            "val": metrics(*predict(model, vl)), "test": metrics(*predict(model, te)),
            "seen": metrics(*predict(model, se)) if se is not None else None,
            "test_last_epoch": last, "history": hist,
            "sizes": {k: len(v) for k, v in data.items()}}


def main():
    args = parse_args()
    data, _ = load_graphs(args)
    split_name = f"to_{args.test_dataset}" if args.test_dataset and args.test_dataset != args.dataset else args.split
    name = f"{args.dataset}_{args.feature}_{split_name}_{args.model}{'_cond' if args.adv_conditional else ''}{args.tag}"
    print(f"[{name}] sizes " + str({k: len(v) for k, v in data.items()}))
    for seed in args.seeds:
        t0 = time.time()
        r = run_seed(args, data, seed)
        r["run_name"] = name
        with open(os.path.join(RUN_DIR, f"{name}_s{seed}.json"), "w") as f:
            json.dump(r, f)
        t = r["test"]
        print(f"  seed {seed}: ep {r['best_epoch']:>2} val_f1 {r['val']['macro_f1']:.3f} | test acc {t['accuracy']:.3f} "
              f"f1 {t['macro_f1']:.3f} auc {t['auc']:.3f} worst_env {t['worst_env_acc']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
