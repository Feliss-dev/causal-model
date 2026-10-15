"""
04_upfd_comparison.py
=====================
Bảng so sánh mô hình trên UPFD theo đúng khuôn của results/final_tables.md (Fakeddit):
Seen Acc | OOD Acc | OOD F1 | OOD AUC | F1 Drop% | Worst-group, mean ± std qua 5 seed.

  Bảng 1  In-distribution (chia gốc UPFD 20/10/70) — so với benchmark UPFD
  Bảng 2  Temporal OOD: train trên tin cũ, test trên 30% tin mới nhất (dịch chuyển tự nhiên)
  Bảng 3  Cross-dataset: train GossipCop -> test PolitiFact và ngược lại
          (lối tắt "kích thước đồ thị" đảo chiều giữa hai bộ -> phép thử confounder thật)

Cách chạy:  uv run python propagation/04_upfd_comparison.py   (sau 02_train_upfd.py)
Output:     results/upfd/comparison_tables.{md,json}, results/upfd/comparison_ood_f1.png
"""

import os
import glob
import json
import numpy as np

RUN_DIR = os.path.join("results", "upfd", "runs")
OUT = os.path.join("results", "upfd")
ROWS = [  # (khoá, tên hiển thị)
    ("root_mlp", "MLP (chỉ nội dung tin)"),
    ("stats_mlp", "MLP (chỉ thống kê cấu trúc)"),
    ("gcn", "GCN (UPFD)"),
    ("gat", "GAT (UPFD)"),
    ("sage", "ERM GraphSAGE (UPFD)"),
    ("irm", "IRM"),
    ("eerm", "EERM"),
    ("causal", "CausalPropGNN (GRL)"),
    ("causal_cond", "CausalPropGNN (GRL có điều kiện)"),
]
OURS = {"causal", "causal_cond"}


def load():
    runs = {}
    for f in glob.glob(os.path.join(RUN_DIR, "*.json")):
        r = json.load(open(f))
        c = r["config"]
        if c["feature"] != "bert":
            continue
        model = c["model"] + ("_cond" if c.get("adv_conditional") else "")
        if c.get("test_dataset") and c["test_dataset"] != c["dataset"]:
            proto = f"cross:{c['dataset']}>{c['test_dataset']}"
        else:
            proto = f"{c['split']}:{c['dataset']}"
        runs.setdefault(proto, {}).setdefault(model, []).append(r)
    return runs


def ms(v, pct=True, digits=1):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    if pct:
        return f"{np.mean(v)*100:.{digits}f} ± {np.std(v)*100:.{digits}f}"
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def best_marks(vals):
    """Đánh dấu giá trị trung bình cao nhất trong cột."""
    means = {k: np.mean(v) for k, v in vals.items() if v}
    return max(means, key=means.get) if means else None


def ood_table(title, runs, note):
    L = [f"### {title}", "", note, "",
         "| Mô hình | Seen Acc % | OOD Acc % | OOD Macro-F1 % | OOD AUC | F1 Drop % | Worst-group % | Sụp đổ |",
         "|---|---|---|---|---|---|---|---|"]
    out, cols = {}, {"f1": {}, "auc": {}, "wg": {}}
    for key, name in ROWS:
        rs = runs.get(key)
        if not rs or rs[0].get("seen") is None:
            continue
        seen_acc = [r["seen"]["accuracy"] for r in rs]
        acc = [r["test"]["accuracy"] for r in rs]
        f1 = [r["test"]["macro_f1"] for r in rs]
        auc = [r["test"]["auc"] for r in rs]
        drop = [(r["seen"]["macro_f1"] - r["test"]["macro_f1"]) / r["seen"]["macro_f1"]
                for r in rs if r["seen"]["macro_f1"] > 0]
        wg = [r["test"]["worst_group_acc"] for r in rs]
        coll = sum(r["test"]["collapsed"] for r in rs)
        out[key] = {"seen_acc": seen_acc, "ood_acc": acc, "ood_f1": f1, "ood_auc": auc,
                    "f1_drop": drop, "worst_group": wg, "collapsed": coll, "n": len(rs)}
        cols["f1"][key], cols["auc"][key], cols["wg"][key] = f1, auc, [w for w in wg if w is not None]
    best = {c: best_marks(v) for c, v in cols.items()}
    for key, name in ROWS:
        if key not in out:
            continue
        o = out[key]
        f1 = ms(o["ood_f1"]); auc = ms(o["ood_auc"], False); wg = ms(o["worst_group"])
        f1 = f"**{f1}**" if best["f1"] == key else f1
        auc = f"**{auc}**" if best["auc"] == key else auc
        wg = f"**{wg}**" if best["wg"] == key else wg
        label = f"**{name}**" if key in OURS else name
        L.append(f"| {label} | {ms(o['seen_acc'])} | {ms(o['ood_acc'])} | {f1} | {auc} | "
                 f"{ms(o['f1_drop'])} | {wg} | {o['collapsed']}/{o['n']} |")
    L.append("")
    return L, out


def id_table(runs_by_proto):
    L = ["### Bảng 1 — In-distribution (chia gốc UPFD 20/10/70, feature `bert`)", "",
         "Đây là thiết lập benchmark gốc của UPFD: train/val/test ngẫu nhiên cùng phân phối.", "",
         "| Mô hình | PolitiFact Acc % | PolitiFact F1 % | PolitiFact AUC | GossipCop Acc % | GossipCop F1 % | GossipCop AUC |",
         "|---|---|---|---|---|---|---|"]
    out = {}
    for key, name in ROWS:
        cells = []
        for ds in ["politifact", "gossipcop"]:
            rs = runs_by_proto.get(f"original:{ds}", {}).get(key)
            if not rs:
                cells += ["—"] * 3
                continue
            t = [r["test"] for r in rs]
            out[f"{ds}|{key}"] = {k: [x[k] for x in t] for k in ["accuracy", "macro_f1", "auc"]}
            cells += [ms([x["accuracy"] for x in t]), ms([x["macro_f1"] for x in t]),
                      ms([x["auc"] for x in t], False)]
        if any(c != "—" for c in cells):
            label = f"**{name}**" if key in OURS else name
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    return L, out


def chart(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f2328", "#5f6670", "#e3e5e8"
    BASE, OURS_C = "#9aa3ad", "#2a78d6"
    panels = [(k, t) for k, t in [("temporal:politifact", "Temporal OOD · PolitiFact"),
                                  ("temporal:gossipcop", "Temporal OOD · GossipCop"),
                                  ("cross:gossipcop>politifact", "Cross · GossipCop → PolitiFact"),
                                  ("cross:politifact>gossipcop", "Cross · PolitiFact → GossipCop")]
              if k in results]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    names = [n for k, n in ROWS]
    for ax, (k, title) in zip(axes, panels):
        res = results[k]
        ys = np.arange(len(ROWS))[::-1]
        for y, (key, name) in zip(ys, ROWS):
            if key not in res:
                continue
            v = np.array(res[key]["ood_f1"]) * 100
            c = OURS_C if key in OURS else BASE
            ax.barh(y, v.mean(), height=0.62, color=c, edgecolor="white", linewidth=2)
            ax.errorbar(v.mean(), y, xerr=v.std(), color=INK, linewidth=1, capsize=2.5)
            ax.text(v.mean() + v.std() + 1.5, y, f"{v.mean():.1f}", va="center", fontsize=8.5, color=INK)
        ax.axvline(50, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 105)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("OOD macro-F1 (%)", color=MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=MUTED, length=0)
    axes[0].set_yticks(np.arange(len(ROWS))[::-1])
    axes[0].set_yticklabels(names, fontsize=9, color=INK)
    fig.text(0.01, 0.01, "Thanh xanh = mô hình nhân quả (CausalPropGNN); xám = baseline. "
             "Vạch đứt = 50%. Sai số = ±1 std qua 5 seed.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(OUT, "comparison_ood_f1.png")
    fig.savefig(path, dpi=160, facecolor="white")
    print(f"Saved {path}")


def main():
    runs = load()
    L = ["# UPFD — bảng so sánh mô hình (khuôn giống Fakeddit `final_tables.md`)", "",
         "Mean ± std qua 5 seed, feature `bert`. Nhãn 1 = Fake. Checkpoint chọn theo val macro-F1 "
         "(val cùng phân phối với train, không dùng test). **Đậm** ở cột = trung bình tốt nhất. "
         "Seen = test cùng phân phối (15% giữ lại từ giai đoạn/bộ dữ liệu train). "
         "Worst-group = độ chính xác thấp nhất giữa 4 nhóm (đồ thị nhỏ/lớn × Fake/Real). "
         "\"Sụp đổ\" = macro-F1 < 50% hoặc > 90% dự đoán một lớp.", ""]
    t1, o1 = id_table(runs)
    L += t1
    results = {"in_distribution": o1}
    L += ["### Bảng 2 — Temporal OOD (dịch chuyển thời gian tự nhiên)", ""]
    for ds in ["politifact", "gossipcop"]:
        k = f"temporal:{ds}"
        if k in runs:
            t, o = ood_table(f"2.{1 if ds == 'politifact' else 2} — {ds}", runs[k],
                             "Train trên 70% tin cũ nhất (chia 75/10/15 train/val/seen), test trên 30% tin mới nhất.")
            L += t
            results[k] = o
    L += ["### Bảng 3 — Cross-dataset (lối tắt cấu trúc đảo chiều)", ""]
    for a, b in [("gossipcop", "politifact"), ("politifact", "gossipcop")]:
        k = f"cross:{a}>{b}"
        if k in runs:
            t, o = ood_table(f"3.{1 if a == 'gossipcop' else 2} — {a} → {b}", runs[k],
                             f"Train/val/seen từ {a}, test (OOD) = toàn bộ {b}.")
            L += t
            results[k] = o
    with open(os.path.join(OUT, "comparison_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "comparison_tables.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(L))
    chart(results)


if __name__ == "__main__":
    main()
