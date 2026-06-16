"""
make_neo4j_bi_figures.py
========================
Sinh các hình minh họa cho phần Neo4j (biểu diễn đồ thị) và BI dashboard,
từ dữ liệu thật trong data/processed. Lưu vào Fair_Article/figures/ + copy VN.

Chạy: PYTHONUTF8=1 uv run python make_neo4j_bi_figures.py
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

D = os.path.join("data", "processed")
OUT_EN = os.path.join("Fair_Article", "figures")
OUT_VN = os.path.join("Fair_Article_VN", "figures")
os.makedirs(OUT_EN, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})


def _csvlen(f):
    p = os.path.join(D, f)
    return len(pd.read_csv(p)) if os.path.exists(p) else 0


# ===================== FIG: Neo4j HIN schema =====================
def fig_schema():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Neo4j Heterogeneous Information Network — schema",
                 fontsize=13, fontweight="bold")

    # node positions
    nodes = {
        "Post\n(X)":      (5.0, 4.0, "#2e7d32"),
        "User\n(C₂)":     (2.0, 6.3, "#d1495b"),
        "Subreddit\n(C₁)":(2.0, 1.7, "#d1495b"),
        "Domain\n(D)":    (8.0, 6.3, "#4c72b0"),
        "Image\n(I)":     (8.0, 1.7, "#edae49"),
    }
    R = 0.85
    for label, (x, y, c) in nodes.items():
        ax.add_patch(Circle((x, y), R, color=c, ec="black", lw=1.3, zorder=3))
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)

    edges = [
        ("Post\n(X)", "User\n(C₂)",      "POSTED_BY"),
        ("Post\n(X)", "Subreddit\n(C₁)", "POSTED_IN"),
        ("Post\n(X)", "Domain\n(D)",     "LINKS_TO"),
        ("Post\n(X)", "Image\n(I)",      "HAS_IMAGE"),
        ("User\n(C₂)", "Subreddit\n(C₁)","MEMBER_OF"),
    ]
    for a, b, lab in edges:
        xa, ya, _ = nodes[a]; xb, yb, _ = nodes[b]
        dx, dy = xb - xa, yb - ya
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        p1 = (xa + ux * R, ya + uy * R)
        p2 = (xb - ux * R, yb - uy * R)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                     color="#555", lw=1.6, zorder=2))
        mx, my = (xa + xb) / 2, (ya + yb) / 2
        ax.text(mx, my, lab, ha="center", va="center", fontsize=8.5,
                color="#222", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaa", lw=0.6),
                zorder=5)

    # legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#2e7d32", label="Causal content (X)"),
           Patch(facecolor="#d1495b", label="Confounders (C₁, C₂)"),
           Patch(facecolor="#4c72b0", label="Source factor (D)"),
           Patch(facecolor="#edae49", label="Visual factor (I)")]
    ax.legend(handles=leg, loc="upper center", ncol=2, fontsize=8.5,
              bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_EN, "fig_neo4j_schema.png"))
    plt.close(fig)


# ===================== FIG: graph statistics =====================
def fig_graph_stats():
    nodes = {
        "Post": _csvlen("posts.csv"), "User": _csvlen("users.csv"),
        "Subreddit": _csvlen("subreddits.csv"), "Domain": _csvlen("domains.csv"),
        "Image": _csvlen("images.csv"),
    }
    edges = {
        "POSTED_BY": _csvlen("posted_by.csv"), "POSTED_IN": _csvlen("posted_in.csv"),
        "LINKS_TO": _csvlen("links_to.csv"), "HAS_IMAGE": _csvlen("has_image.csv"),
        "MEMBER_OF": _csvlen("member_of.csv"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    nc = "#4c72b0"; ec = "#66a182"
    b1 = axes[0].bar(list(nodes.keys()), list(nodes.values()), color=nc,
                     edgecolor="black", lw=0.5)
    axes[0].set_title(f"Node counts (total = {sum(nodes.values()):,})")
    axes[0].set_ylabel("Count")
    for b in b1:
        axes[0].annotate(f"{int(b.get_height()):,}",
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                         xytext=(0, 2), textcoords="offset points")
    axes[0].tick_params(axis="x", rotation=20)

    b2 = axes[1].bar(list(edges.keys()), list(edges.values()), color=ec,
                     edgecolor="black", lw=0.5)
    axes[1].set_title(f"Relationship counts (total = {sum(edges.values()):,})")
    axes[1].set_ylabel("Count")
    for b in b2:
        axes[1].annotate(f"{int(b.get_height()):,}",
                         (b.get_x() + b.get_width() / 2, b.get_height()),
                         ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                         xytext=(0, 2), textcoords="offset points")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Fakeddit HIN stored in Neo4j", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_EN, "fig_graph_stats.png"))
    plt.close(fig)


# ===================== FIG: BI dashboard composite =====================
def fig_bi_overview():
    posts = pd.read_csv(os.path.join(D, "posts_enriched.csv"))
    domains = pd.read_csv(os.path.join(D, "domains_enriched.csv"))
    subs = pd.read_csv(os.path.join(D, "subreddits_enriched.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    # (a) Louvain community sizes (top 6)
    ax = axes[0, 0]
    if "community_id" in posts.columns:
        vc = posts["community_id"].value_counts().head(6)
        ax.pie(vc.values, labels=[f"C{c}" for c in vc.index], autopct="%1.0f%%",
               colors=plt.cm.Set3.colors, wedgeprops=dict(ec="white", lw=1))
        ax.set_title("(a) Louvain communities (Post nodes, top 6)")
    else:
        ax.axis("off")

    # (b) Top fake-rate domains (post_count>=3 to avoid singletons)
    ax = axes[0, 1]
    dd = domains[domains["post_count"] >= 3].sort_values(
        "fake_ratio_real", ascending=False).head(8)
    ax.barh(dd["url_domain"][::-1], dd["fake_ratio_real"][::-1],
            color="#d1495b", edgecolor="black", lw=0.5)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Fake ratio")
    ax.set_title("(b) Top misinformation domains (≥3 posts)")
    ax.tick_params(axis="y", labelsize=8)

    # (c) Subreddit fake-rate distribution
    ax = axes[1, 0]
    ss = subs.sort_values("fake_ratio_real", ascending=False)
    colors = ["#d1495b" if v > 0.5 else "#2e7d32" for v in ss["fake_ratio_real"]]
    ax.bar(range(len(ss)), ss["fake_ratio_real"], color=colors,
           edgecolor="black", lw=0.4)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("Subreddit (ranked)"); ax.set_ylabel("Community fake ratio")
    ax.set_title(f"(c) Pure-label communities ({len(ss)} subreddits)")
    ax.text(len(ss) * 0.5, 0.93, "near-0/1 ratios ⇒ subreddit defines label",
            fontsize=8.5, style="italic", color="#444", ha="center")

    # (d) User spreader profile: post_count vs avg_score colored by fake_rate
    ax = axes[1, 1]
    users = pd.read_csv(os.path.join(D, "users_enriched.csv"))
    sc = ax.scatter(users["avg_score"], users["post_count"],
                    c=users.get("fake_rate", 0), cmap="RdYlGn_r",
                    s=18, alpha=0.6, edgecolor="none")
    ax.set_xlabel("Avg post score"); ax.set_ylabel("Post count")
    ax.set_title("(d) User spreader profiles")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("user fake rate", fontsize=9)

    fig.suptitle("BI dashboard panels — descriptive graph analytics (Neo4j GDS + Plotly)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_EN, "fig_bi_overview.png"))
    plt.close(fig)


def main():
    fig_schema()
    fig_graph_stats()
    fig_bi_overview()
    os.makedirs(OUT_VN, exist_ok=True)
    for f in ["fig_neo4j_schema.png", "fig_graph_stats.png", "fig_bi_overview.png"]:
        shutil.copy(os.path.join(OUT_EN, f), os.path.join(OUT_VN, f))
    print("Saved Neo4j + BI figures to", OUT_EN, "and copied to", OUT_VN)


if __name__ == "__main__":
    main()
