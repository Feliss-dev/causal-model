"""
Hình 2 — Kiến trúc CausalHeteroGNN (MÃ GỐC đã tạo ảnh trong tài liệu).

Chạy:  python fig2_architecture.py
Kết quả: figures/image2.png
"""
import os
os.makedirs("figures", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# Colors
C_INPUT   = "#1F3A5F"   # dark blue - input / main path
C_SPUR    = "#C25E12"   # orange - spurious branch
C_CAUSAL  = "#2E5A1C"   # dark green - causal branch
C_ENCODER = "#2C3E55"   # slate - shared encoder
C_GRL     = "#7B2FBE"   # purple - GRL / adversarial
C_CONF    = "#7A3E12"   # brown - confounder classifier
C_MAIN    = "#1F3A5F"   # dark blue - main classifier
TXT = "white"

fig, ax = plt.subplots(figsize=(15.5, 10.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

def box(x, y, w, h, color, lines, fs=15, weight="bold", tcolor=TXT, sub=None, subfs=12):
    """Draw a rounded box centered at (x,y) with multi-line text."""
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(b)
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    gap = 2.9  # data-unit line spacing (roomy for Vietnamese diacritics)
    for i, ln in enumerate(lines):
        yy = y + ((n-1)/2 - i) * gap
        ax.text(x, yy, ln, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=tcolor, zorder=4)
    return b

def arrow(x1, y1, x2, y2, color, lw=3.0, style="-|>", rad=0.0, ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=26,
                        linewidth=lw, color=color, zorder=2,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)
    return a

def label(x, y, text, color, fs=12.5, bg="white", weight="bold"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight,
            color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc=bg, ec=color, lw=1.3))

# ---------- Title ----------
ax.text(50, 98, "Hình 2.  Kiến trúc CausalHeteroGNN", ha="center", va="center",
        fontsize=20, fontweight="bold", color="#1F3A5F")
ax.text(50, 94.2,
        "Dự đoán chính: h_causal → Bộ phân loại Fake/Real    |    "
        "GRL + L_adv khử tín hiệu cộng đồng    |    L_ortho tách hai biểu diễn",
        ha="center", va="center", fontsize=12.5, color="#444444")

# ---------- Nodes ----------
# Input
box(50, 86, 42, 10, C_INPUT,
    ["HeteroData Input", "(Post / User / Subreddit / Domain / Image)"], fs=15)

# Two graphs
box(23, 70, 36, 10.5, C_SPUR,
    ["G  —  Đồ thị dị thể đầy đủ", "(giữ nguyên mọi quan hệ)"], fs=14.5)
box(77, 70, 36, 10.5, C_CAUSAL,
    ["G_causal  —  Đồ thị can thiệp", "(xóa cạnh liên quan Subreddit)"], fs=14.5)

# Shared encoder
box(50, 53, 64, 10, C_ENCODER,
    ["Heterogeneous GraphSAGE Encoder", "(dùng chung trọng số,  d = 96)"], fs=15.5)

# Representations
box(23, 37, 34, 10, C_SPUR,
    ["h_spurious", "(biểu diễn phụ trợ)"], fs=14.5)
box(77, 37, 34, 10, C_CAUSAL,
    ["h_causal", "(biểu diễn nhân quả)"], fs=14.5)

# GRL
box(50, 30.0, 18, 8.5, C_GRL, ["GRL", "(α = 2.0)"], fs=14)

# Classifiers (bottom)
box(20, 14.5, 34, 10, C_CONF,
    ["Bộ phân loại", "Subreddit (Confounder)"], fs=13.5)
box(80, 14.5, 36, 10, C_MAIN,
    ["Bộ phân loại Fake / Real", "(Đầu ra chính → ŷ)"], fs=14)

# ---------- Arrows ----------
# input -> graphs
arrow(42, 83, 28, 75, C_SPUR)
arrow(58, 83, 72, 75, C_CAUSAL)
# graphs -> encoder
arrow(27, 65.3, 42, 57.2, C_SPUR)
arrow(73, 65.3, 58, 57.2, C_CAUSAL)
# encoder -> representations
arrow(38, 49.3, 27, 41.6, C_SPUR)
arrow(62, 49.3, 73, 41.6, C_CAUSAL)
# h_spurious -> confounder classifier
arrow(22, 32.4, 20.8, 19.6, C_SPUR)
# h_causal -> main classifier
arrow(78, 32.4, 79.4, 19.6, C_MAIN)
# h_causal -> GRL  (curved)
arrow(64.5, 34.5, 58.5, 31.5, C_GRL, rad=-0.25)
# GRL -> confounder classifier (adversarial)
arrow(42, 28.5, 30, 19.6, C_GRL, rad=-0.18)

# L_ortho dashed double arrow between representations
arrow(39, 37, 61, 37, C_GRL, lw=2.4, style="<|-|>", ls=(0,(6,4)))
label(50, 40.4, "L_ortho  (ràng buộc trực giao)", C_GRL, fs=12)

# ---------- Loss labels ----------
label(10.5, 24, "L_spurious", C_CONF, fs=11.5)
label(50, 22.0, "L_adv", C_GRL, fs=11.5)
label(92.5, 24, "L_causal", C_MAIN, fs=11.5)

# ---------- Legend ----------
legend_items = [
    ("Nhánh phụ trợ  (G đầy đủ)", C_SPUR),
    ("Nhánh nhân quả  (G_causal)", C_CAUSAL),
    ("Đường dự đoán chính", C_MAIN),
    ("GRL / mất mát đối kháng", C_GRL),
    ("Bộ phân loại Confounder", C_CONF),
]
handles = [Line2D([0],[0], color=c, lw=8) for _, c in legend_items]
labels  = [t for t, _ in legend_items]
leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=3, fontsize=12.5, frameon=True, handlelength=1.6,
                columnspacing=1.8, borderpad=0.8)
leg.get_frame().set_edgecolor("#CCCCCC")

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.07)
plt.savefig("figures/image2.png", dpi=200,
            bbox_inches="tight", facecolor="#F2F4F7")
print("Figure 2 saved")
