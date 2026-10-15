"""
16_mediaeval_diag_report.py
===========================
Tổng hợp các lần chạy chẩn đoán (results/mediaeval_diag/runs/*.json, do 15_train_mediaeval.py
ghi) thành bảng mean ± std qua seed và vẽ đường cong huấn luyện cho A, B, C.

Cách chạy:  uv run python pipeline/16_mediaeval_diag_report.py
Output:     results/mediaeval_diag/summary.{md,json}, results/mediaeval_diag/curves_ABC.png
"""

import os
import glob
import json
import numpy as np
import torch

DIAG_DIR = os.path.join("results", "mediaeval_diag")
RUN_DIR = os.path.join(DIAG_DIR, "runs")
GRAPH_PATH = os.path.join("data", "mediaeval2015", "graph", "graph.pt")

DESCRIPTIONS = {
    "M":  "MLP content-only (mốc sàn, không đồ thị)",
    "A":  "Baseline GNN (chỉ nhánh G, chỉ L_base_2w)",
    "full": "CausalHeteroGNN đầy đủ (α = 2, hằng)",
    "B":  "CausalHeteroGNN, α = 0",
    "C1": "CausalHeteroGNN, lịch DANN α_max = 1",
    "C2": "CausalHeteroGNN, lịch DANN α_max = 2",
    "D":  "CausalHeteroGNN, không node Image",
    "E":  "CausalHeteroGNN, class weight",
    "F":  "Kết hợp (xem cấu hình)",
    "full_b": "full, chạy lại (val Boston) để lấy kết quả epoch cuối",
    "M_v2": "M, val = Boston + Sochi",
    "A_v2": "A, val = Boston + Sochi",
    "full_v2": "full, val = Boston + Sochi",
}
ORDER = ["M", "A", "full", "B", "C1", "C2", "D", "E", "full_b", "M_v2", "A_v2", "full_v2", "F"]
CURVE_RUNS = ["A", "full", "B", "C1", "C2"]
# Bảng màu categorical tham chiếu (thứ tự cố định, không xoay vòng)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#1f2328", "#5f6670", "#e3e5e8"


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RUN_DIR, "*.json"))):
        r = json.load(open(f))
        runs.setdefault(r["run_name"], []).append(r)
    return runs


def ms(vals, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    m, s = np.mean(vals), np.std(vals)
    return f"{m*100:.1f} ± {s*100:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


def summary(runs):
    names = [n for n in ORDER if n in runs] + sorted(n for n in runs if n not in ORDER)
    L = ["# Chẩn đoán CausalHeteroGNN trên MediaEval 2015", "",
         "Mean ± std qua seed. Test = testset MediaEval 2015 (3.781 tweet, 67,8% Fake). "
         "Val = Boston. AUC dùng P(Fake), Fake là lớp dương. "
         "\"Sụp đổ\" = macro-F1 < 0,5 hoặc > 90% dự đoán rơi vào một lớp.", "",
         "| Run | Cấu hình | Acc % | Macro-F1 % | AUC | Recall Fake % | Recall Real % | Dự đoán Fake % | Sụp đổ | Val macro-F1 % | Epoch tốt nhất |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for n in names:
        rs = runs[n]
        t = [r["test"] for r in rs]
        row = {k: [x[k] for x in t] for k in ["accuracy", "macro_f1", "auc", "recall_fake",
                                                 "recall_real", "pred_share_fake"]}
        row["collapsed"] = sum(x["collapsed"] for x in t)
        row["val_macro_f1"] = [r["val"]["macro_f1"] for r in rs]
        row["best_epoch"] = [r["best_epoch"] for r in rs]
        row["cm_mean"] = np.mean([x["confusion_matrix[true Fake,Real][pred Fake,Real]"] for x in t], axis=0).round(0).tolist()
        out[n] = row
        L.append(f"| {n} | {DESCRIPTIONS.get(n, n)} | {ms(row['accuracy'], 1)} | {ms(row['macro_f1'], 1)} | "
                 f"{ms(row['auc'])} | {ms(row['recall_fake'], 1)} | {ms(row['recall_real'], 1)} | "
                 f"{ms(row['pred_share_fake'], 1)} | {row['collapsed']}/{len(rs)} | "
                 f"{ms(row['val_macro_f1'], 1)} | {np.mean(row['best_epoch']):.0f} |")
    L += ["", "## Ma trận nhầm lẫn trung bình trên test", "",
          "| Run | Fake → Fake | Fake → Real | Real → Fake | Real → Real |", "|---|---|---|---|---|"]
    for n in names:
        (ff, fr), (rf, rr) = out[n]["cm_mean"]
        L.append(f"| {n} | {ff:.0f} | {fr:.0f} | {rf:.0f} | {rr:.0f} |")
    last = [n for n in names if all("test_last_epoch" in r for r in runs[n])]
    if last:
        L += ["", "## Chẩn đoán: mô hình ở epoch cuối (không chọn checkpoint)", "",
              "Chỉ để đọc, không dùng test để chọn gì. So sánh với bảng trên để tách ảnh hưởng của việc chọn checkpoint.", "",
              "| Run | Acc % | Macro-F1 % | AUC | Dự đoán Fake % | Sụp đổ |", "|---|---|---|---|---|---|"]
        for n in last:
            t = [r["test_last_epoch"] for r in runs[n]]
            L.append(f"| {n} | {ms([x['accuracy'] for x in t], 1)} | {ms([x['macro_f1'] for x in t], 1)} | "
                     f"{ms([x['auc'] for x in t])} | {ms([x['pred_share_fake'] for x in t], 1)} | "
                     f"{sum(x['collapsed'] for x in t)}/{len(t)} |")
    with open(os.path.join(DIAG_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(DIAG_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(L))


def curves(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = torch.load(GRAPH_PATH, weights_only=False)
    conf = g["conf_label"][g["train_mask"]]
    majority = float(torch.bincount(conf).max() / len(conf))

    panels = [("val_macro_f1", "Val macro-F1 (Boston)", None),
              ("L_base_2w", "L_base_2w (train)", None),
              ("L_adv", "L_adv (train)", None),
              ("conf_acc_h_causal", "Độ chính xác confounder_clf trên h_causal", majority),
              ("conf_acc_h_spurious", "Độ chính xác confounder_clf trên h_spurious", majority),
              ("cos_abs", "mean |cos(h_c, h_s)|", None)]
    names = [n for n in CURVE_RUNS if n in runs]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (key, title, ref) in zip(axes.flat, panels):
        for c, n in zip(COLORS, names):
            series = [r["history"][key] for r in runs[n]]
            if all(v is None for s in series for v in s):
                continue
            arr = np.array([[np.nan if v is None else v for v in s] for s in series], dtype=float)
            ep = np.arange(1, arr.shape[1] + 1)
            m, s = np.nanmean(arr, 0), np.nanstd(arr, 0)
            ax.fill_between(ep, m - s, m + s, color=c, alpha=0.12, linewidth=0)
            ax.plot(ep, m, color=c, linewidth=2, label=n)
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1, linestyle="--")
            ax.text(ax.get_xlim()[1], ref, f"  tỉ lệ sự kiện lớn nhất {ref:.2f}", color=MUTED,
                    fontsize=8.5, va="bottom", ha="right")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("epoch")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, [f"{l}: {DESCRIPTIONS[l]}" for l in labels], loc="upper center",
               ncol=3, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(DIAG_DIR, "curves_ABC.png")
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Saved {path}")


if __name__ == "__main__":
    runs = load_runs()
    summary(runs)
    curves(runs)
