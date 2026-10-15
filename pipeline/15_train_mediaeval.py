"""
15_train_mediaeval.py
=====================
Huấn luyện + đánh giá CausalHeteroGNN trên đồ thị MediaEval 2015 (từ 14_mediaeval_graph.py)
cho loạt thí nghiệm chẩn đoán A–F. Kiến trúc giữ nguyên (lớp CausalHeteroGNN của
05_train_gnn.py); Event là confounder, Tweet là node cần phân loại.

Khác với 05 (theo đặc tả chẩn đoán):
  - Không có nhãn 6 lớp -> bỏ L_base_6w, L_causal_6w.
  - Val = sự kiện Boston (tách từ devset). Scheduler + chọn checkpoint theo val macro-F1.
  - Train cố định --epochs epoch (mặc định 200), không early stop, để lịch α kiểu DANN
    chạy hết p ∈ [0, 1] và đường cong các lần chạy so sánh được với nhau.
  - Full-batch, transductive (tweet val/test có trong đồ thị, loss chỉ trên train).

Cờ:
  --model {causal,baseline,mlp}   baseline = lần chạy A (chỉ nhánh G, chỉ L_base_2w)
  --grl_alpha A                   α của GRL (B: 0 -> encoder không nhận gradient đối kháng)
  --grl_schedule {const,dann}     dann: α(p) = α_max·(2/(1+exp(−10p)) − 1), p = epoch/epochs
  --no_image_nodes                bỏ node Image + has_image; nối CLIP 512 vào đặc trưng Tweet
  --class_weight                  CE có trọng số nghịch tần suất lớp cho các loss 2-way
  --seeds 0 1 2 3 4  --epochs 200  --run_name A

Ví dụ:
  uv run python pipeline/15_train_mediaeval.py --run_name A --model baseline
  uv run python pipeline/15_train_mediaeval.py --run_name B --grl_alpha 0
Output: results/mediaeval_diag/runs/{run_name}_s{seed}.json
"""

import os
import sys
import json
import copy
import time
import argparse
import importlib
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
train_gnn = importlib.import_module("05_train_gnn")
erm_mlp = importlib.import_module("08_baselines_erm_mlp")

GRAPH_PATH = os.path.join("data", "mediaeval2015", "graph", "graph.pt")
RUN_DIR = os.path.join("results", "mediaeval_diag", "runs")
os.makedirs(RUN_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument("--model", choices=["causal", "baseline", "mlp"], default="causal")
    p.add_argument("--grl_alpha", type=float, default=train_gnn.GRL_ALPHA)
    p.add_argument("--grl_schedule", choices=["const", "dann"], default="const")
    p.add_argument("--no_image_nodes", action="store_true")
    p.add_argument("--class_weight", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val_events", nargs="+", default=["boston"],
                   help="sự kiện dev dùng làm val (mặc định Boston, như graph.pt)")
    return p.parse_args()


# ===================== DATA =====================

def resplit(g, val_events):
    """Chia lại train/val theo danh sách sự kiện val; tính lại nhãn confounder."""
    import pandas as pd
    meta = pd.read_csv(os.path.join(os.path.dirname(GRAPH_PATH), "tweets_meta.csv"), dtype=str)
    is_test = meta["split"].eq("test").values
    is_val = meta["event"].isin(val_events).values & ~is_test
    is_train = ~is_test & ~is_val
    train_events = sorted(meta.loc[is_train, "event"].unique())
    idx = {e: k for k, e in enumerate(train_events)}
    conf = [idx[e] if t else -1 for e, t in zip(meta["event"], is_train)]
    g["train_mask"], g["val_mask"] = torch.tensor(is_train), torch.tensor(is_val)
    g["conf_label"] = torch.tensor(conf, dtype=torch.long)
    g["train_events"] = train_events
    return g


def build_data(no_image_nodes, with_mlp_feats=False, val_events=("boston",)):
    g = torch.load(GRAPH_PATH, weights_only=False)
    if sorted(val_events) != ["boston"]:
        g = resplit(g, list(val_events))
    data = HeteroData()
    for t, x in g["x"].items():
        if no_image_nodes and t == "Image":
            continue
        data[t].x = x
    if no_image_nodes or with_mlp_feats:
        data["Tweet"].x = torch.cat([g["x"]["Tweet"], g["tweet_clip_mean"]], dim=1)
    for et, ei in g["edges"].items():
        if no_image_nodes and "Image" in (et[0], et[2]):
            continue
        data[et].edge_index = ei
    data = T.ToUndirected()(data)
    return data, g


# ===================== HELPERS =====================

def grl_alpha_at(args, epoch):
    if args.grl_schedule == "const":
        return args.grl_alpha
    p = epoch / args.epochs
    return args.grl_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def test_metrics(y, logits):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, recall_score
    y = y.numpy()
    pred = logits.argmax(1).numpy()
    p_fake = F.softmax(logits, dim=1)[:, 0].numpy()
    share = np.bincount(pred, minlength=2) / len(pred)
    m = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(roc_auc_score(1 - y, p_fake)) if len(set(y)) == 2 else None,
        "recall_fake": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "pred_share_fake": float(share[0]),
        "pred_share_real": float(share[1]),
        "confusion_matrix[true Fake,Real][pred Fake,Real]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    m["collapsed"] = bool(m["macro_f1"] < 0.5 or share.max() > 0.9)
    return m


def macro_f1(y, logits):
    from sklearn.metrics import f1_score
    return float(f1_score(y.numpy(), logits.argmax(1).numpy(), average="macro", zero_division=0))


# ===================== ONE RUN =====================

def run_seed(args, data, g, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    y = g["y"]
    tr, va, te = g["train_mask"], g["val_mask"], g["test_mask"]
    conf = g["conf_label"]
    n_events = len(g["train_events"])
    hidden, dropout = train_gnn.HIDDEN_CHANNELS, train_gnn.DROPOUT

    if args.model == "mlp":
        mlp = erm_mlp.MLPContentOnly(data["Tweet"].x.shape[1], hidden, dropout)
        model = mlp
        fwd = lambda: mlp({"Post": data["Tweet"].x})
    else:
        dims = {t: data[t].x.shape[1] for t in data.node_types}
        model = train_gnn.CausalHeteroGNN(
            data.metadata(), hidden_channels=hidden, num_subreddits=n_events, dropout=dropout,
            grl_alpha=args.grl_alpha, edge_dropout=train_gnn.EDGE_DROPOUT, causal_cut=True,
            node_feat_dims=dims, confounder_type="Event", target_type="Tweet")
        fwd = lambda: model(data.x_dict, data.edge_index_dict)

    # class weight nghịch tần suất trên train (chỉ khi --class_weight)
    weight = None
    if args.class_weight:
        cnt = torch.bincount(y[tr], minlength=2).float()
        weight = cnt.sum() / (2.0 * cnt)

    opt = torch.optim.Adam(model.parameters(), lr=train_gnn.LR, weight_decay=train_gnn.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=10, min_lr=1e-5)
    hist = {k: [] for k in ["epoch", "loss", "L_base_2w", "L_causal_2w", "L_spurious", "L_adv",
                            "L_ortho", "alpha", "lr", "conf_acc_h_causal", "conf_acc_h_spurious",
                            "cos_abs", "val_macro_f1", "val_acc"]}
    best_f1, best_state, best_epoch = -1.0, None, 0

    def predict(out):
        if args.model == "mlp":
            return out
        return out[0] if args.model == "baseline" else out[1]

    for epoch in range(1, args.epochs + 1):
        alpha = grl_alpha_at(args, epoch)
        if args.model != "mlp":
            model.grl.alpha = alpha
        model.train()
        opt.zero_grad()
        out = fwd()
        comp = {}
        if args.model == "mlp":
            loss = F.cross_entropy(out[tr], y[tr], weight=weight)
            comp["L_base_2w"] = loss
        else:
            pb2, pc2, _, _, sub_s, sub_c, h_c, h_s = out
            comp["L_base_2w"] = F.cross_entropy(pb2[tr], y[tr], weight=weight)
            if args.model == "baseline":
                loss = comp["L_base_2w"]
            else:
                comp["L_causal_2w"] = F.cross_entropy(pc2[tr], y[tr], weight=weight)
                comp["L_spurious"] = F.cross_entropy(sub_s[tr], conf[tr])
                comp["L_adv"] = F.cross_entropy(sub_c[tr], conf[tr])
                comp["L_ortho"] = torch.mean(torch.abs(F.cosine_similarity(h_c[tr], h_s[tr])))
                loss = (comp["L_base_2w"] + comp["L_causal_2w"] + 0.5 * comp["L_spurious"]
                        + train_gnn.W_SUB_ADV * comp["L_adv"] + 0.2 * comp["L_ortho"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            ev = fwd()
            logits = predict(ev)
            vf1 = macro_f1(y[va], logits[va])
            vacc = float((logits[va].argmax(1) == y[va]).float().mean())
            if args.model == "causal":
                _, _, _, _, s_s, s_c, e_hc, e_hs = ev
                ca_c = float((s_c[tr].argmax(1) == conf[tr]).float().mean())
                ca_s = float((s_s[tr].argmax(1) == conf[tr]).float().mean())
                cos = float(torch.abs(F.cosine_similarity(e_hc[tr], e_hs[tr])).mean())
            else:
                ca_c = ca_s = cos = None
        sched.step(vf1)

        hist["epoch"].append(epoch); hist["loss"].append(loss.item())
        for k in ["L_base_2w", "L_causal_2w", "L_spurious", "L_adv", "L_ortho"]:
            hist[k].append(comp[k].item() if k in comp else None)
        hist["alpha"].append(float(alpha)); hist["lr"].append(opt.param_groups[0]["lr"])
        hist["conf_acc_h_causal"].append(ca_c); hist["conf_acc_h_spurious"].append(ca_s)
        hist["cos_abs"].append(cos); hist["val_macro_f1"].append(vf1); hist["val_acc"].append(vacc)

        if vf1 > best_f1:
            best_f1, best_epoch = vf1, epoch
            best_state = copy.deepcopy(model.state_dict())

    # Chẩn đoán: mô hình ở epoch cuối (không chọn checkpoint) — chỉ để đọc, không dùng chọn gì
    model.eval()
    with torch.no_grad():
        test_last = test_metrics(y[te], predict(fwd())[te])
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = predict(fwd())
    return {
        "run_name": args.run_name, "seed": seed, "config": vars(args),
        "n_train_events": n_events, "best_epoch": best_epoch,
        "val": test_metrics(y[va], logits[va]),
        "test": test_metrics(y[te], logits[te]),
        "test_last_epoch": test_last,
        "history": hist,
    }


def main():
    args = parse_args()
    data, g = build_data(args.no_image_nodes, with_mlp_feats=(args.model == "mlp"),
                         val_events=args.val_events)
    print(f"[{args.run_name}] {vars(args)}")
    print(f"  node types {data.node_types} | Tweet dim {data['Tweet'].x.shape[1]} | "
          f"edge types {len(data.edge_types)} | train events {g['train_events']}")
    for seed in args.seeds:
        t0 = time.time()
        res = run_seed(args, data, g, seed)
        with open(os.path.join(RUN_DIR, f"{args.run_name}_s{seed}.json"), "w") as f:
            json.dump(res, f)
        t = res["test"]
        print(f"  seed {seed}: best_ep {res['best_epoch']:>3} val_f1 {res['val']['macro_f1']:.3f} | "
              f"test acc {t['accuracy']:.3f} f1 {t['macro_f1']:.3f} auc {t['auc']:.3f} "
              f"predFake {t['pred_share_fake']:.2f} collapsed={t['collapsed']} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
