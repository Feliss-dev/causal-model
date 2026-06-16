"""
Hình 1 — Sơ đồ property-graph của HIN đa phương thức trong Neo4j.

LƯU Ý: Đây là bản DỰNG LẠI khớp với ảnh trong tài liệu gốc (số liệu node/cạnh
đọc từ ảnh và mô tả dataset Fakeddit). Không phải mã gốc đã sinh ảnh ban đầu.

Chạy:  python fig1_neo4j_schema.py
Kết quả: figures/image1.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

# (tên node, x, y, màu, dòng mô tả, số lượng)
nodes = {
    "Subreddit": (0.50, 0.82, "#E0392B", "C\u2081 \u2013 Confounder", "n = 21"),
    "User":      (0.82, 0.58, "#F39C12", "C\u2082 \u2013 Confounder", "n = 4,604"),
    "Post":      (0.50, 0.50, "#27AE60", "X \u2013 Content",     "n = 5,898"),
    "Domain":    (0.18, 0.55, "#2E86DE", "D \u2013 Source",      "n = 658"),
    "Image":     (0.50, 0.18, "#8E44AD", "I \u2013 Visual",      "n = 5,898"),
}

# (node_a, node_b, nhãn quan hệ, lệch nhãn theo (dx, dy))
edges = [
    ("Subreddit", "Post", "POSTED_IN",  (0.06, 0.0)),
    ("User",      "Subreddit", "MEMBER_OF", (0.02, 0.04)),
    ("Post",      "User", "POSTED_BY",  (0.04, 0.04)),
    ("Domain",    "Post", "LINKS_TO",   (0.0, 0.03)),
    ("Post",      "Image", "HAS_IMAGE", (0.12, 0.0)),
]

fig, ax = plt.subplots(figsize=(10.5, 8), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("Property-graph schema of the multimodal HIN in Neo4j",
             fontsize=17, fontweight="bold", pad=16)

R = 0.085  # bán kính node
# vẽ cạnh trước
for a, b, lbl, (dx, dy) in edges:
    xa, ya = nodes[a][0], nodes[a][1]
    xb, yb = nodes[b][0], nodes[b][1]
    ax.plot([xa, xb], [ya, yb], color="#555555", lw=2, zorder=1)
    mx, my = (xa + xb) / 2 + dx, (ya + yb) / 2 + dy
    ax.text(mx, my, lbl, fontsize=11, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888888", lw=1),
            zorder=3)

# vẽ node
for name, (x, y, color, desc, cnt) in nodes.items():
    circ = plt.Circle((x, y), R, color=color, zorder=2)
    ax.add_patch(circ)
    ax.text(x, y, name, color="white", fontsize=13, fontweight="bold",
            ha="center", va="center", zorder=4)
    if name == "Subreddit":
        # đặt mô tả phía trên node (node nằm trên đỉnh sơ đồ)
        ax.text(x, y + R + 0.085, desc, fontsize=11, style="italic",
                color="#333333", ha="center", va="bottom")
        ax.text(x, y + R + 0.04, cnt, fontsize=10, color="#777777",
                ha="center", va="bottom")
    else:
        # đặt mô tả phía dưới node
        ax.text(x, y - R - 0.04, desc, fontsize=11, style="italic",
                color="#333333", ha="center", va="top")
        ax.text(x, y - R - 0.085, cnt, fontsize=10, color="#777777",
                ha="center", va="top")

ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("figures/image1.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved figures/image1.png")
