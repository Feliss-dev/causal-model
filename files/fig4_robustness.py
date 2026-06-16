"""
Hình 4 — Phân tích robustness: Worst-Group Accuracy & Label-Flip Rate.

Panel (a): đọc từ results/worst_group_conf.json (sinh bởi 09_worst_group.py).
Panel (b): đọc từ results/metrics_bd_s42.json — LFR của seed 42 chạy không
           skip-explain (xem run_all.ps1 phase main). Fallback về hằng số nếu
           file chưa có trường lfr.

Chạy:  python files/fig4_robustness.py
Kết quả: figures/image5.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# ---------- Panel (a): Worst-Group ----------
WG_FILE  = os.path.join("results", "worst_group_conf.json")
with open(WG_FILE, encoding="utf-8") as f:
    wg_data = json.load(f)["summary"]

# Thứ tự models: key trong JSON → nhãn trục x
WG_MODEL_MAP = [
    ("MLP",            "MLP\ncontent-only"),
    ("ERM",            "Baseline\nGNN"),
    ("IRM",            "IRM"),
    ("EERM",           "EERM"),
    ("CausalHeteroGNN","CausalHeteroGNN"),
]

models    = [label for _, label in WG_MODEL_MAP]
colors    = ["#3B6FB6", "#E8821E", "#4C9A2A", "#F4C20D", "#C0392B"]

worst     = [wg_data[k]["worst_group_acc"][0] * 100 for k, _ in WG_MODEL_MAP]
worst_err = [wg_data[k]["worst_group_acc"][1] * 100 for k, _ in WG_MODEL_MAP]
avg       = [wg_data[k]["avg_group_acc"][0]   * 100 for k, _ in WG_MODEL_MAP]
avg_err   = [wg_data[k]["avg_group_acc"][1]   * 100 for k, _ in WG_MODEL_MAP]

# ---------- Panel (b): LFR ----------
# Đọc từ metrics_bd_s42.json (seed 42, confounding-shift, không skip-explain)
LFR_METRICS = os.path.join("results", "metrics_bd_s42.json")
if os.path.exists(LFR_METRICS):
    with open(LFR_METRICS, encoding="utf-8") as f:
        mets = json.load(f)
    if "lfr" in mets:
        lfr = mets["lfr"]
        # Thứ tự: [subreddit_swap, remove_image, change_domain]
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

# ---------- Vẽ ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=200)
fig.suptitle("Figure 4.  Robustness Analysis: Worst-Group Accuracy and Label-Flip Rate",
             fontsize=17, fontweight="bold", y=0.99)

# ---- Panel (a) ----
x = np.arange(len(models)); w = 0.38
ax1.bar(x - w/2, worst, w, yerr=worst_err, capsize=5, color=colors,
        edgecolor="black", linewidth=0.6, label="Worst-Group Acc")
ax1.bar(x + w/2, avg,   w, yerr=avg_err,   capsize=5, color=colors, alpha=0.45,
        edgecolor="black", linewidth=0.6, hatch="..", label="Avg-Group Acc")
ax1.axhline(max(worst), color="#C0392B", ls="--", lw=1.4, alpha=0.7)
for xi, v, e in zip(x - w/2, worst, worst_err):
    ax1.text(xi, v + e + 1.5, f"{v:.1f}%", ha="center", va="bottom",
             fontsize=12, fontweight="bold")
ax1.set_title("(a) Worst-Group vs Avg-Group Accuracy\n(Confounding-Shift)",
              fontsize=14, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=13)
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=11)
ax1.set_ylim(0, 95)
ax1.tick_params(axis="y", labelsize=11)
hand = [plt.Rectangle((0, 0), 1, 1, color="#555555"),
        plt.Rectangle((0, 0), 1, 1, color="#555555", alpha=0.45, hatch="..")]
ax1.legend(hand, ["Worst-Group Acc", "Avg-Group Acc"], fontsize=12, loc="upper left")
ax1.grid(axis="y", ls=":", alpha=0.5)

# ---- Panel (b) ----
inter = ["do(C₁=swap)\nSubreddit Swap", "do(I=∅)\nRemove Image",
         "do(D=credible)\nChange Source"]
xi = np.arange(len(inter)); wb = 0.38
ax2.bar(xi - wb/2, base_lfr,   wb, color="#E8821E", edgecolor="black",
        linewidth=0.6, label="Baseline GNN")
ax2.bar(xi + wb/2, causal_lfr, wb, color="#C0392B", edgecolor="black",
        linewidth=0.6, label="CausalHeteroGNN")
for x0, v in zip(xi - wb/2, base_lfr):
    ax2.text(x0, v + 0.6, f"{v:.1f}%", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#B5650E")
for x0, v in zip(xi + wb/2, causal_lfr):
    txt = "~0.0%" if v < 0.1 else f"{v:.1f}%"
    ax2.text(x0, v + 0.6, txt, ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#C0392B")
ax2.set_title("(b) Label-Flip Rate under\nStructural Interventions do(·)",
              fontsize=14, fontweight="bold")
ax2.set_ylabel("Label-Flip Rate (%)", fontsize=13)
ax2.set_xticks(xi); ax2.set_xticklabels(inter, fontsize=11)
ax2.set_ylim(0, 40)
ax2.tick_params(axis="y", labelsize=11)
ax2.legend(fontsize=12, loc="upper right")
ax2.grid(axis="y", ls=":", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figures/image5.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved figures/image5.png")
