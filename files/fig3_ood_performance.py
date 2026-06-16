"""
Hình 3 — Hiệu năng phân loại trên hai giao thức OOD
(Held-Out-Subreddit và Confounding-Shift).

Đọc số liệu thật từ results/final_tables.json (sinh bởi 10_final_tables.py).

Chạy:  python files/fig3_ood_performance.py
Kết quả: figures/image3.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# ---------- Load data từ pipeline JSON ----------
FINAL_TABLES = os.path.join("results", "final_tables.json")
with open(FINAL_TABLES, encoding="utf-8") as f:
    tables = json.load(f)

std_rows  = tables["standard_ood"]
conf_rows = tables["confounding_shift"]

# Thứ tự models cần hiển thị (key trong JSON → nhãn trục x)
MODEL_MAP = [
    ("MLP (content-only)", "MLP\ncontent-only"),
    ("ERM HeteroSAGE",     "Baseline\nGNN"),
    ("IRM",                "IRM"),
    ("EERM",               "EERM"),
    ("CausalHeteroGNN",    "CausalHeteroGNN"),
]

def _get(rows, model_key, field, idx=0, scale=1.0):
    for r in rows:
        if r["model"] == model_key:
            return r[field][idx] * scale
    raise KeyError(f"Model '{model_key}' / field '{field}' not found in JSON")

models    = [label for _, label in MODEL_MAP]
mkeys     = [k     for k, _     in MODEL_MAP]

seen      = [_get(std_rows,  k, "seen_acc", 0, 100) for k in mkeys]
seen_err  = [_get(std_rows,  k, "seen_acc", 1, 100) for k in mkeys]
held      = [_get(std_rows,  k, "ood_acc",  0, 100) for k in mkeys]
held_err  = [_get(std_rows,  k, "ood_acc",  1, 100) for k in mkeys]
conf      = [_get(conf_rows, k, "ood_acc",  0, 100) for k in mkeys]
conf_err  = [_get(conf_rows, k, "ood_acc",  1, 100) for k in mkeys]
auc       = [_get(conf_rows, k, "ood_auc",  0)       for k in mkeys]
f1drop    = [_get(conf_rows, k, "f1_drop",  0)       for k in mkeys]

# ---------- Vẽ ----------
base_colors = ["#3B6FB6", "#E8821E", "#4C9A2A", "#F4C20D", "#C0392B"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=200)
fig.suptitle("Figure 3.  Classification Performance on Two OOD Protocols:\n"
             "Held-Out-Subreddit and Confounding-Shift",
             fontsize=17, fontweight="bold", y=1.0)

# ---- Panel (a): 3 nhóm bar cho mỗi mô hình ----
x = np.arange(len(models)); w = 0.26

def shade(hexc, factor):
    hexc = hexc.lstrip("#")
    r, g, b = (int(hexc[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"

c_seen = [shade(c, 0.45) for c in base_colors]

ax1.bar(x - w, seen, w, yerr=seen_err, capsize=3, color=c_seen,
        edgecolor="black", linewidth=0.4, label="Seen Accuracy")
ax1.bar(x, held, w, yerr=held_err, capsize=3, color=base_colors,
        edgecolor="black", linewidth=0.4, label="Held-Out OOD Acc")
ax1.bar(x + w, conf, w, yerr=conf_err, capsize=3, color=base_colors,
        edgecolor="black", linewidth=0.4, hatch="//", label="Conf-Shift OOD Acc")
ax1.axhline(80, color="#C0392B", ls="--", lw=1.3, alpha=0.7)

for xi, v, e in zip(x + w, conf, conf_err):
    ax1.text(xi, v + e + 1.2, f"{v:.1f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax1.set_title("(a) Accuracy: Seen vs Held-Out vs Confounding-Shift OOD",
              fontsize=13.5, fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 105)
ax1.legend(fontsize=11, loc="upper right")
ax1.grid(axis="y", ls=":", alpha=0.4)

# ---- Panel (b): AUC bars + F1-drop line ----
ax2.bar(x, auc, 0.55, color=base_colors, edgecolor="black", linewidth=0.5)
for xi, v in zip(x, auc):
    ax2.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax2.set_ylabel("AUC", fontsize=12, color="#1F4E96")
ax2.set_ylim(0, 1.08)
ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=10)
ax2.set_title("(b) AUC & F1-drop on Confounding-Shift",
              fontsize=13.5, fontweight="bold")

ax2b = ax2.twinx()
ax2b.plot(x, f1drop, "D--", color="#7B2FBE", lw=2.2, markersize=9,
          label="F1-drop (%)")
for xi, v in zip(x, f1drop):
    ax2b.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                  xytext=(12, 4), fontsize=11, fontweight="bold", color="#7B2FBE")
ax2b.set_ylabel("F1-drop (%)", fontsize=12, color="#7B2FBE")
ax2b.set_ylim(0, 55)

h1 = [plt.Rectangle((0, 0), 1, 1, color="#1F4E96")]
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, ["AUC"] + l2, fontsize=11, loc="upper center")
ax2.grid(axis="y", ls=":", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("figures/image3.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved figures/image3.png")
