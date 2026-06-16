"""
08_baselines_erm_mlp.py
=======================
Hai baseline ĐỘC LẬP bổ sung cho bảng so sánh chính:

  • ERM (Empirical Risk Minimization): HeteroGraphSAGE thuần — đúng backbone
    của CausalHeteroGNN nhưng KHÔNG structural cut, KHÔNG GRL, KHÔNG ortho,
    KHÔNG multi-task, huấn luyện ĐỘC LẬP bằng cross-entropy 2-way.
    → Đây mới là "Baseline GNN" đúng nghĩa cho paper. Con số baseline cũ
      trong metrics_*.json là NHÁNH baseline bên trong CausalHeteroGNN
      (chung encoder, chịu gradient từ causal/adv/ortho loss) nên không phải
      một baseline độc lập về mặt phương pháp luận.

  • MLP content-only: phân loại trực tiếp trên feature của Post
    (mpnet 768 + scalar 3 — KHÔNG message passing, KHÔNG graph).
    → Trả lời câu hỏi phản biện "graph đóng góp gì?" và là điểm neo (anchor)
      content-only cho cả hai protocol.

THIẾT KẾ AN TOÀN — giống 07_baselines_irm_eerm.py:
  - Chỉ đọc dữ liệu qua build_heterodata() của 05_train_gnn.py.
  - Không ghi đè bất kỳ kết quả nào hiện có.
  - Kết quả ghi riêng: results/baselines_erm_mlp{RUN_TAG}.json
    (cùng schema overall/seen/unseen/f1_drop_pct).

Env vars (giống các script khác):
  GNN_INPUT_DIR        : data/processed (mặc định) | data/processed_confounded
  GNN_SEED, GNN_RUN_TAG
  GNN_OOD_TRANSDUCTIVE : "1" cho confounding-shift benchmark

Cách chạy:
  # Standard OOD (inductive content-only)
  uv run python pipeline/08_baselines_erm_mlp.py

  # Confounding-shift (transductive)
  $env:GNN_INPUT_DIR="data/processed_confounded"; $env:GNN_OOD_TRANSDUCTIVE="1"
  uv run python pipeline/08_baselines_erm_mlp.py
"""

import os
import json
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

train_gnn = importlib.import_module("05_train_gnn")
baselines6 = importlib.import_module("07_baselines_irm_eerm")
build_heterodata = train_gnn.build_heterodata
compute_metrics = train_gnn.compute_metrics
GNNClassifier = baselines6.GNNClassifier
evaluate = baselines6.evaluate          # tôn trọng GNN_OOD_TRANSDUCTIVE

# ===================== CONFIG =====================
SEED = int(os.environ.get("GNN_SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)

HIDDEN     = int(os.environ.get("GNN_HIDDEN", "96"))
DROPOUT    = float(os.environ.get("GNN_DROPOUT", "0.4"))
LR         = float(os.environ.get("GNN_LR", "0.005"))
WD         = float(os.environ.get("GNN_WD", "5e-4"))
MAX_EPOCHS = int(os.environ.get("GNN_MAX_EPOCHS", "300"))
PATIENCE   = int(os.environ.get("GNN_PATIENCE", "30"))
RUN_TAG    = os.environ.get("GNN_RUN_TAG", "")

OUTPUT_DIR = "results"
MODEL_DIR  = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class MLPContentOnly(nn.Module):
    """Phân loại 2-way chỉ từ feature của Post node — không dùng graph.
    Nhận (x_dict, edge_index_dict) như các model khác để dùng chung evaluate()."""

    def __init__(self, post_feat_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_feat_dim, hidden), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),        nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_dict, edge_index_dict=None):
        return self.net(x_dict["Post"])


def train_simple(model, data, train_mask, val_mask, device, name, ckpt_name):
    """Vòng train chung: CE 2-way, early stop theo val loss — y hệt 03/06."""
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10)
    y = data["Post"].y
    best_val, no_improve = float("inf"), 0
    ckpt = os.path.join(MODEL_DIR, ckpt_name)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); opt.zero_grad()
        logits = model(data.x_dict, data.edge_index_dict)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
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
            print(f"  [{name}] ep{epoch:03d} loss={loss.item():.3f} "
                  f"vloss={vloss:.3f} vf1={vf1:.3f} pat={no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print(f"  [{name}] early stop @ ep{epoch}")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def main():
    data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, \
        user_feat_dim, domain_feat_dim, post_feat_dim = build_heterodata()

    device = torch.device("cpu")
    transductive = os.environ.get("GNN_OOD_TRANSDUCTIVE", "0") == "1"
    print(f"\nDevice: {device} | seed={SEED} | transductive_eval={transductive}")
    data = data.to(device)
    train_mask = data["Post"].train_mask
    val_mask   = data["Post"].val_mask

    print("\n" + "=" * 60 + "\n  TRAIN ERM (standalone HeteroGraphSAGE)\n" + "=" * 60)
    erm = GNNClassifier(data.metadata(), HIDDEN, DROPOUT, post_feat_dim,
                        user_feat_dim, domain_feat_dim).to(device)
    n_params_erm = sum(p.numel() for p in erm.parameters())
    print(f"  params: {n_params_erm:,}")
    erm = train_simple(erm, data, train_mask, val_mask, device,
                       "ERM", f"erm{RUN_TAG}.pt")
    erm_metrics = evaluate(erm, data, posts_df, device)

    print("\n" + "=" * 60 + "\n  TRAIN MLP (content-only, no graph)\n" + "=" * 60)
    mlp = MLPContentOnly(post_feat_dim, HIDDEN, DROPOUT).to(device)
    n_params_mlp = sum(p.numel() for p in mlp.parameters())
    print(f"  params: {n_params_mlp:,}")
    mlp = train_simple(mlp, data, train_mask, val_mask, device,
                       "MLP", f"mlp{RUN_TAG}.pt")
    mlp_metrics = evaluate(mlp, data, posts_df, device)

    out = {
        "_config": {
            "seed": SEED, "transductive_eval": transductive,
            "input_dir": os.environ.get("GNN_INPUT_DIR", "data/processed"),
            "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "wd": WD,
            "params_erm": n_params_erm, "params_mlp": n_params_mlp,
            "note": ("ERM = standalone HeteroGraphSAGE (same backbone as "
                     "CausalHeteroGNN, no cut/GRL/ortho/multi-task). "
                     "MLP = content-only Post-feature classifier, no graph."),
        },
        "erm": erm_metrics,
        "mlp": mlp_metrics,
    }
    out_path = os.path.join(OUTPUT_DIR, f"baselines_erm_mlp{RUN_TAG}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4, default=str)
    print(f"\nĐã lưu {out_path}")

    print("\n" + "=" * 64)
    print(f"  {'Model':<14}{'OOD Acc':>10}{'OOD F1':>10}{'OOD AUC':>10}{'F1 Drop%':>10}")
    print("  " + "-" * 50)
    for name, m in [("ERM", erm_metrics), ("MLP", mlp_metrics)]:
        u = m["unseen"]
        print(f"  {name:<14}{u['accuracy']:>10.4f}{u['f1']:>10.4f}"
              f"{u['auc']:>10.4f}{m['f1_drop_pct']:>10.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
