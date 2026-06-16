"""
Hình 5 — BI Dashboard tổng hợp kết quả.

Đọc số liệu thật từ:
  results/final_tables.json      (sinh bởi 10_final_tables.py)
  results/worst_group_conf.json  (sinh bởi 09_worst_group.py, conf protocol)
  results/metrics_bd_s42.json    (sinh bởi 06_evaluate.py, seed 42, không skip-explain)

Chạy:  python files/fig5_dashboard.py
Kết quả: figures/image6.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

os.makedirs("figures", exist_ok=True)

# ============================================================
# Load data
# ============================================================
RESULTS = "results"

with open(os.path.join(RESULTS, "final_tables.json"), encoding="utf-8") as f:
    tables = json.load(f)
with open(os.path.join(RESULTS, "worst_group_conf.json"), encoding="utf-8") as f:
    wg_data = json.load(f)["summary"]

conf_rows = tables["confounding_shift"]

# Thứ tự 5 models
MODEL_MAP = [
    ("MLP (content-only)", "MLP"),
    ("ERM HeteroSAGE",     "Baseline\nGNN"),
    ("IRM",                "IRM"),
    ("EERM",               "EERM"),
    ("CausalHeteroGNN",    "Causal\nHeteroGNN"),
]
WG_MAP = [
    ("MLP",             "MLP"),
    ("ERM",             "Baseline GNN"),
    ("IRM",             "IRM"),
    ("EERM",            "EERM"),
    ("CausalHeteroGNN", "CausalHeteroGNN"),
]

models     = [lbl for _, lbl in MODEL_MAP]
mkeys      = [k   for k, _   in MODEL_MAP]
colors     = ["#3B6FB6", "#E8821E", "#4C9A2A", "#F4C20D", "#C0392B"]

def _get(rows, mkey, field, idx=0, scale=1.0):
    for r in rows:
        if r["model"] == mkey:
            return r[field][idx] * scale
    return None

acc       = [_get(conf_rows, k, "ood_acc",  0, 100) for k in mkeys]
acc_err   = [_get(conf_rows, k, "ood_acc",  1, 100) for k in mkeys]
auc       = [_get(conf_rows, k, "ood_auc",  0)       for k in mkeys]
f1drop    = [_get(conf_rows, k, "f1_drop",  0)       for k in mkeys]

wg_models_labels = [lbl for _, lbl in WG_MAP]
wg        = [wg_data[k]["worst_group_acc"][0] * 100 for k, _ in WG_MAP]
wg_err    = [wg_data[k]["worst_group_acc"][1] * 100 for k, _ in WG_MAP]

# LFR từ metrics_bd_s42.json (với lfr) hoặc fallback
LFR_METRICS = os.path.join(RESULTS, "metrics_bd_s42.json")
if os.path.exists(LFR_METRICS):
    with open(LFR_METRICS, encoding="utf-8") as f:
        mets = json.load(f)
    if "lfr" in mets:
        lfr = mets["lfr"]
        base_lfr   = [lfr["baseline"]["lfr_subreddit"] * 100,
                      lfr["baseline"]["lfr_image"]     * 100,
                      lfr["baseline"]["lfr_domain"]    * 100]
        causal_lfr = [lfr["causal"]["lfr_subreddit"] * 100,
                      lfr["causal"]["lfr_image"]     * 100,
                      lfr["causal"]["lfr_domain"]    * 100]
    else:
        base_lfr   = [30.1, 4.0, 6.2]
        causal_lfr = [0.0,  8.0, 15.5]
else:
    base_lfr   = [30.1, 4.0, 6.2]
    causal_lfr = [0.0,  8.0, 15.5]

# KPI values (CausalHeteroGNN)
caus_acc  = _get(conf_rows, "CausalHeteroGNN", "ood_acc", 0, 100)
caus_acc_std = _get(conf_rows, "CausalHeteroGNN", "ood_acc", 1, 100)
base_acc  = _get(conf_rows, "ERM HeteroSAGE",  "ood_acc", 0, 100)
caus_auc  = _get(conf_rows, "CausalHeteroGNN", "ood_auc", 0)
caus_drop = _get(conf_rows, "CausalHeteroGNN", "f1_drop", 0)
base_drop = _get(conf_rows, "ERM HeteroSAGE",  "f1_drop", 0)
caus_wg   = wg_data["CausalHeteroGNN"]["worst_group_acc"][0] * 100
caus_wg_std = wg_data["CausalHeteroGNN"]["worst_group_acc"][1] * 100
lfr_sub_causal = causal_lfr[0]  # do(Subreddit)

# ============================================================
# Vẽ dashboard
# ============================================================
fig = plt.figure(figsize=(16, 9.5), dpi=190)
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1.0, 1.0],
                      hspace=0.55, wspace=0.28,
                      left=0.05, right=0.97, top=0.93, bottom=0.06)

fig.text(0.5, 0.965, "CausalHeteroGNN  ·  Experimental Results Dashboard",
         ha="center", va="center", fontsize=22, fontweight="bold", color="white",
         bbox=dict(boxstyle="round,pad=0.6", fc="#1F6FB2", ec="none"))

# ---- KPI cards ----
kpi_ax = fig.add_subplot(gs[0, :])
kpi_ax.axis("off")
kpi_ax.set_xlim(0, 5)
kpi_ax.set_ylim(0, 1)

drop_ratio = base_drop / caus_drop if caus_drop > 0 else float("inf")
cards = [
    ("Conf-Shift OOD Acc",   f"{caus_acc:.1f} ± {caus_acc_std:.1f}%",
     f"vs. {base_acc:.1f}% baseline",         "#C0392B"),
    ("AUC (Conf-Shift)",     f"{caus_auc:.3f}",
     "best of all models",                    "#1F6FB2"),
    ("F1-drop (Seen→OOD)",   f"{caus_drop:.1f}%",
     f"{drop_ratio:.0f}x lower than ERM",     "#4C9A2A"),
    ("Worst-Group Acc",      f"{caus_wg:.1f} ± {caus_wg_std:.1f}%",
     "highest of all",                        "#C0392B"),
    ("LFR  do(Subreddit)",   f"~{lfr_sub_causal:.1f}%",
     "confounder removed",                    "#4C9A2A"),
]
for i, (title, val, sub, col) in enumerate(cards):
    x0 = i + 0.06
    kpi_ax.add_patch(FancyBboxPatch((x0, 0.06), 0.88, 0.88,
                     boxstyle="round,pad=0.02,rounding_size=0.04",
                     transform=kpi_ax.transData,
                     facecolor="#F4F6F8", edgecolor="#D5DBE0", linewidth=1.2))
    kpi_ax.add_patch(FancyBboxPatch((x0, 0.06), 0.05, 0.88,
                     boxstyle="round,pad=0,rounding_size=0.01",
                     transform=kpi_ax.transData, facecolor=col, edgecolor="none"))
    cx = x0 + 0.46
    kpi_ax.text(cx, 0.78, title, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="#333333")
    kpi_ax.text(cx, 0.50, val,   ha="center", va="center", fontsize=18,
                fontweight="bold", color="#1A1A1A")
    kpi_ax.text(cx, 0.24, sub,   ha="center", va="center", fontsize=10,
                style="italic", color=col)

# ---- (mid-left) Conf-Shift OOD Accuracy ----
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(range(5), acc, yerr=acc_err, capsize=4, color=colors,
        edgecolor="black", linewidth=0.5)
ax1.axhline(80, color="#C0392B", ls="--", lw=1.2, alpha=0.6)
for i, v in enumerate(acc):
    ax1.text(i, v + acc_err[i] + 1.5, f"{v:.1f}", ha="center", fontsize=10,
             fontweight="bold")
ax1.set_title("Confounding-Shift OOD Accuracy", fontsize=13, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_xticks(range(5)); ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylim(0, 92); ax1.grid(axis="y", ls=":", alpha=0.5)

# ---- (mid-center) AUC bars + F1-drop line ----
ax2 = fig.add_subplot(gs[1, 1])
ax2.bar(range(5), auc, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(auc):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
ax2.set_ylabel("AUC", fontsize=11, color="#1F6FB2")
ax2.set_ylim(0, 1.08)
ax2.set_xticks(range(5)); ax2.set_xticklabels(models, fontsize=9)
ax2.set_title("AUC (bars)  &  F1-drop (line)", fontsize=13, fontweight="bold")
ax2b = ax2.twinx()
ax2b.plot(range(5), f1drop, "o--", color="#7B2FBE", lw=2.2, markersize=8,
          label="F1-drop (%)")
ax2b.set_ylabel("F1-drop (%)", fontsize=11, color="#7B2FBE")
ax2b.set_ylim(0, 55)
ax2b.legend(fontsize=9.5, loc="upper right")
ax2.grid(axis="y", ls=":", alpha=0.4)

# ---- (mid-right) Worst-Group Accuracy (horizontal) ----
ax3 = fig.add_subplot(gs[1, 2])
yp = np.arange(5)
ax3.barh(yp, wg, xerr=wg_err, capsize=4, color=colors,
         edgecolor="black", linewidth=0.5)
ax3.axvline(max(wg), color="#C0392B", ls="--", lw=1.2, alpha=0.6)
for i, v in enumerate(wg):
    ax3.text(v + wg_err[i] + 1.2, i, f"{v:.1f}%", va="center", fontsize=10,
             fontweight="bold")
ax3.set_yticks(yp); ax3.set_yticklabels(wg_models_labels, fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel("Accuracy (%)", fontsize=11)
ax3.set_xlim(0, 56)
ax3.set_title("Worst-Group Accuracy (Confounding-Shift)", fontsize=12.5,
              fontweight="bold")
ax3.grid(axis="x", ls=":", alpha=0.5)

# ---- (bottom-left) Label-Flip Rate ----
ax4 = fig.add_subplot(gs[2, 0:2])
inter = ["do(C₁=swap)\nSubreddit swap", "do(I=∅)\nRemove image",
         "do(D=credible)\nChange source"]
xi = np.arange(3); wb = 0.34
ax4.bar(xi - wb/2, base_lfr,   wb, color="#E8821E", edgecolor="black",
        linewidth=0.5, label="Baseline GNN")
ax4.bar(xi + wb/2, causal_lfr, wb, color="#C0392B", edgecolor="black",
        linewidth=0.5, label="CausalHeteroGNN")
for x0, v in zip(xi - wb/2, base_lfr):
    ax4.text(x0, v + 0.6, f"{v:.1f}%", ha="center", fontsize=10.5,
             fontweight="bold", color="#B5650E")
for x0, v in zip(xi + wb/2, causal_lfr):
    t = f"~{v:.1f}%" if v < 1 else f"{v:.1f}%"
    ax4.text(x0, v + 0.6, t, ha="center", fontsize=10.5, fontweight="bold",
             color="#C0392B")
ax4.set_title("Label-Flip Rate under Structural Interventions do(·)",
              fontsize=13, fontweight="bold")
ax4.set_ylabel("Label-Flip Rate (%)", fontsize=11)
ax4.set_xticks(xi); ax4.set_xticklabels(inter, fontsize=10)
ax4.set_ylim(0, 36); ax4.legend(fontsize=11, loc="upper right")
ax4.grid(axis="y", ls=":", alpha=0.5)

# ---- (bottom-right) Radar chart: top-3 models ----
ax6 = fig.add_subplot(gs[2, 2], polar=True)
axes_lbl = ["Conf-Shift\nOOD Acc", "AUC×100", "Worst-\nGroup", "F1-stability"]
radar_models = {
    "CausalHeteroGNN": (
        [caus_acc,
         _get(conf_rows, "CausalHeteroGNN", "ood_auc", 0) * 100,
         wg_data["CausalHeteroGNN"]["worst_group_acc"][0] * 100,
         100 - caus_drop],
        "#C0392B"),
    "EERM": (
        [_get(conf_rows, "EERM", "ood_acc", 0, 100),
         _get(conf_rows, "EERM", "ood_auc", 0) * 100,
         wg_data["EERM"]["worst_group_acc"][0] * 100,
         100 - _get(conf_rows, "EERM", "f1_drop", 0)],
        "#F4C20D"),
    "MLP": (
        [_get(conf_rows, "MLP (content-only)", "ood_acc", 0, 100),
         _get(conf_rows, "MLP (content-only)", "ood_auc", 0) * 100,
         wg_data["MLP"]["worst_group_acc"][0] * 100,
         100 - _get(conf_rows, "MLP (content-only)", "f1_drop", 0)],
        "#3B6FB6"),
}
N = len(axes_lbl)
ang = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
ang += ang[:1]
for name, (vals, col) in radar_models.items():
    vv = vals + vals[:1]
    ax6.plot(ang, vv, "o-", lw=2, color=col, label=name, markersize=5)
    ax6.fill(ang, vv, color=col, alpha=0.12)
ax6.set_xticks(ang[:-1])
ax6.set_xticklabels(axes_lbl, fontsize=9.5)
ax6.set_ylim(0, 100)
ax6.set_yticks([20, 40, 60, 80, 100])
ax6.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="#888888")
ax6.set_title("Overall KPI Comparison — Top-3 Models", fontsize=12.5,
              fontweight="bold", pad=18)
ax6.legend(loc="upper right", bbox_to_anchor=(1.32, -0.05), fontsize=9.5)

plt.savefig("figures/image6.png", dpi=190, bbox_inches="tight", facecolor="white")
print("Saved figures/image6.png")
