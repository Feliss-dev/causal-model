"""
11_worst_group.py
=================
Tính Worst-Group / Average-Group Accuracy trên OOD test cho mọi model từ
checkpoint CÓ SẴN (không train lại).

Group định nghĩa theo (subreddit, label) của OOD posts:
  • Standard OOD:      nhóm = OOD subreddit (theonion=Fake, neutralnews=Real)
                       → group acc ≈ per-community recall; worst-group là
                       con số FAIR/DomainBed reviewer quan tâm nhất sau OOD acc.
  • Confounding-shift: nhóm = (synthetic env × label) — 4 nhóm; worst-group
                       đo model có hi sinh nhóm bị reversed-correlation không.

Cách chạy:
  # Standard OOD
  $env:PYTHONUTF8="1"; uv run python 11_worst_group.py

  # Confounding-shift
  $env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
  uv run python 11_worst_group.py

Output: results/worst_group_{stdood|conf}.json + bảng console.
"""

import os
import json
import importlib
from collections import defaultdict
import numpy as np
import torch

train_gnn = importlib.import_module("05_train_gnn")
b6 = importlib.import_module("07_baselines_irm_eerm")
b7 = importlib.import_module("08_baselines_erm_mlp")

build_heterodata = train_gnn.build_heterodata
CausalHeteroGNN = train_gnn.CausalHeteroGNN
mask_post_edges = b6.mask_post_edges
GNNClassifier = b6.GNNClassifier
MLPContentOnly = b7.MLPContentOnly

MODEL_DIR  = "models"
OUTPUT_DIR = "results"
TRANSDUCTIVE = os.environ.get("GNN_OOD_TRANSDUCTIVE", "0") == "1"
CONF = "confounded" in os.environ.get("GNN_INPUT_DIR", "")
PROTO = "conf" if CONF else "stdood"

HIDDEN  = int(os.environ.get("GNN_HIDDEN", "96"))
DROPOUT = float(os.environ.get("GNN_DROPOUT", "0.4"))

# Checkpoint tags per protocol (seed 42 của IRM/EERM cũ không có hậu tố _s42).
# GNN_USE_CLIPCONS=1 → đánh giá các checkpoint CLIPcons (772-d input) — phải
# build data với cùng flag, nên tách thành bộ tag riêng (PROTO + "_cc").
USE_CC = os.environ.get("GNN_USE_CLIPCONS", "0") == "1"
_seeds5 = ["_s42", "_s1", "_s2", "_s3", "_s4"]
CKPT_TAGS = {
    "stdood": {
        "causal_gnn": ["_main_s42", "_main_s1", "_main_s2"],
        "erm":  ["_stdood_s42", "_stdood_s1", "_stdood_s2"],
        "mlp":  ["_stdood_s42", "_stdood_s1", "_stdood_s2"],
        "irm":  ["", "_stdood_s1", "_stdood_s2"],
        "eerm": ["", "_stdood_s1", "_stdood_s2"],
    },
    "conf": {
        "causal_gnn": ["_bd_s42", "_bd_s1", "_bd_s2"],
        "erm":  ["_conf_s42", "_conf_s1", "_conf_s2"],
        "mlp":  ["_conf_s42", "_conf_s1", "_conf_s2"],
        "irm":  ["_conf", "_conf_s1", "_conf_s2"],
        "eerm": ["_conf", "_conf_s1", "_conf_s2"],
    },
    "stdood_cc": {
        "causal_gnn": [f"_ccmain{s}" for s in _seeds5],
        "erm":  [f"_ccstd{s}" for s in _seeds5],
        "mlp":  [f"_ccstd{s}" for s in _seeds5],
    },
    "conf_cc": {
        "causal_gnn": [f"_ccbd{s}" for s in _seeds5],
        "erm":  [f"_ccconf{s}" for s in _seeds5],
        "mlp":  [f"_ccconf{s}" for s in _seeds5],
    },
}[PROTO + ("_cc" if USE_CC else "")]


def group_metrics(y_true, y_pred, groups):
    """Acc per group + worst/average-group acc."""
    accs = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        accs[g] = float(np.mean(y_true[idx] == y_pred[idx]))
    vals = list(accs.values())
    return {"per_group": accs,
            "worst_group_acc": float(min(vals)),
            "avg_group_acc": float(np.mean(vals)),
            "group_sizes": {g: int(sum(1 for x in groups if x == g))
                            for g in sorted(set(groups))}}


def main():
    data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, \
        user_feat_dim, domain_feat_dim, post_feat_dim = build_heterodata()
    device = torch.device("cpu")
    data = data.to(device)
    num_subreddits = len(subreddits_df)

    test_mask = data["Post"].test_mask
    ood_mask = data["Post"].ood_mask
    ood_idx = torch.where(test_mask & ood_mask)[0]
    y = data["Post"].y

    # Group label per OOD post: (subreddit, true-label)
    pos = posts_df.reset_index(drop=True)
    groups = []
    for i in ood_idx.tolist():
        sub = pos.iloc[i]["subreddit"]
        lbl = "Real" if int(pos.iloc[i]["label_2way"]) == 1 else "Fake"
        groups.append(f"{sub}|{lbl}")
    y_ood = y[ood_idx].numpy()

    print(f"\nProtocol={PROTO} | transductive={TRANSDUCTIVE} | OOD posts={len(ood_idx)}")
    print(f"Groups: {sorted(set(groups))}")

    # Edge dict theo chế độ inference (khớp 04/06)
    if TRANSDUCTIVE:
        infer_edges = data.edge_index_dict
    else:
        test_ids = set(int(i) for i in torch.where(test_mask)[0])
        infer_edges = mask_post_edges(data.edge_index_dict, test_ids)

    def predict(model, head=None):
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, infer_edges)
        if head == "baseline":
            logits = out[0]
        elif head == "causal":
            logits = out[1]
        else:
            logits = out
        return logits[ood_idx].argmax(dim=1).numpy()

    results = {}

    def add(name, preds):
        m = group_metrics(y_ood, preds, groups)
        results.setdefault(name, []).append(m)

    for model_kind, tags in CKPT_TAGS.items():
        for tag in tags:
            ckpt = os.path.join(MODEL_DIR, f"{model_kind}{tag}.pt")
            if not os.path.exists(ckpt):
                print(f"  [skip] {ckpt} không tồn tại")
                continue
            if model_kind == "causal_gnn":
                model = CausalHeteroGNN(
                    data.metadata(), hidden_channels=HIDDEN,
                    num_subreddits=num_subreddits, dropout=DROPOUT,
                    user_feat_dim=user_feat_dim, domain_feat_dim=domain_feat_dim,
                    post_feat_dim=post_feat_dim,
                    grl_alpha=train_gnn.GRL_ALPHA, edge_dropout=0.0,
                    causal_cut=train_gnn.CAUSAL_CUT)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                add("BaselineBranch", predict(model, "baseline"))
                add("CausalHeteroGNN", predict(model, "causal"))
            elif model_kind == "mlp":
                model = MLPContentOnly(post_feat_dim, HIDDEN, DROPOUT)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                add("MLP", predict(model))
            else:  # erm / irm / eerm dùng chung GNNClassifier
                model = GNNClassifier(data.metadata(), HIDDEN, DROPOUT,
                                      post_feat_dim, user_feat_dim, domain_feat_dim)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                add(model_kind.upper(), predict(model))

    # ---- Aggregate over seeds + print ----
    summary = {}
    print(f"\n{'Model':<18}{'WorstGrp':>10}{'AvgGrp':>10}  per-group (mean over seeds)")
    print("-" * 78)
    for name, runs in results.items():
        wg = [r["worst_group_acc"] for r in runs]
        ag = [r["avg_group_acc"] for r in runs]
        per_g = {}
        for g in runs[0]["per_group"]:
            per_g[g] = [float(np.mean([r["per_group"][g] for r in runs])),
                        float(np.std([r["per_group"][g] for r in runs]))]
        summary[name] = {
            "n_seeds": len(runs),
            "worst_group_acc": [float(np.mean(wg)), float(np.std(wg))],
            "avg_group_acc": [float(np.mean(ag)), float(np.std(ag))],
            "per_group": per_g,
            "group_sizes": runs[0]["group_sizes"],
            "per_seed": runs,
        }
        pg = "  ".join(f"{g}={v[0]:.2f}" for g, v in per_g.items())
        print(f"{name:<18}{np.mean(wg):>10.4f}{np.mean(ag):>10.4f}  {pg}")

    out_path = os.path.join(OUTPUT_DIR,
                            f"worst_group_{PROTO}{'_cc' if USE_CC else ''}.json")
    with open(out_path, "w") as f:
        json.dump({"_config": {"protocol": PROTO, "transductive": TRANSDUCTIVE,
                               "groups": sorted(set(groups)),
                               "n_ood": int(len(ood_idx))},
                   "summary": summary}, f, indent=4)
    print(f"\nĐã lưu {out_path}")


if __name__ == "__main__":
    main()
