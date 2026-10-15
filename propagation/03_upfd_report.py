"""
03_upfd_report.py
=================
Tổng hợp results/upfd/runs/*.json (do 02_train_upfd.py ghi) thành bảng mean ± std qua seed.

Cách chạy:  uv run python propagation/03_upfd_report.py
Output:     results/upfd/summary.{md,json}
"""

import os
import glob
import json
import numpy as np

RUN_DIR = os.path.join("results", "upfd", "runs")
OUT = os.path.join("results", "upfd")
MODEL_ORDER = ["root_mlp", "stats_mlp", "gcn", "sage", "gat", "causal", "causal_cond"]
MODEL_NAME = {
    "root_mlp": "MLP trên node tin (không đồ thị)",
    "stats_mlp": "MLP trên thống kê cấu trúc",
    "gcn": "GCN + pool + tin", "sage": "GraphSAGE + pool + tin", "gat": "GAT + pool + tin",
    "causal": "SAGE + GRL môi trường thời gian",
    "causal_cond": "SAGE + GRL có điều kiện theo nhãn",
}


def ms(v, pct=True):
    v = [x for x in v if x is not None]
    if not v:
        return "—"
    return f"{np.mean(v)*100:.1f} ± {np.std(v)*100:.1f}" if pct else f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def main():
    groups = {}
    for f in glob.glob(os.path.join(RUN_DIR, "*.json")):
        r = json.load(open(f))
        c = r["config"]
        model = c["model"] + ("_cond" if c.get("adv_conditional") else "")
        proto = (f"{c['dataset']} → {c['test_dataset']}" if c.get("test_dataset") and c["test_dataset"] != c["dataset"]
                 else f"{c['dataset']} · {c['split']}")
        groups.setdefault((proto, c["feature"]), {}).setdefault(model, []).append(r)

    L = ["# UPFD — kết quả phân loại đồ thị lan truyền", "",
         "Mean ± std qua 5 seed. Nhãn 1 = Fake. Checkpoint chọn theo val macro-F1; cột \"Epoch cuối\" là "
         "macro-F1 của mô hình ở epoch cuối (không chọn checkpoint) để kiểm tra độ nhạy với việc chọn. "
         "\"Môi trường tệ nhất\" = độ chính xác thấp nhất giữa các khoảng thời gian.", ""]
    out = {}
    for (proto, feat) in sorted(groups):
        L += [f"## {proto} — feature `{feat}`", "",
              "| Mô hình | Acc % | Macro-F1 % | AUC | Môi trường tệ nhất % | Dự đoán Fake % | Sụp đổ | Epoch cuối: Macro-F1 % |",
              "|---|---|---|---|---|---|---|---|"]
        g = groups[(proto, feat)]
        for m in [x for x in MODEL_ORDER if x in g]:
            rs = g[m]
            t = [r["test"] for r in rs]
            tl = [r["test_last_epoch"] for r in rs]
            row = {k: [x[k] for x in t] for k in ["accuracy", "macro_f1", "auc", "worst_env_acc", "pred_share_fake"]}
            row["collapsed"] = sum(x["collapsed"] for x in t)
            row["last_macro_f1"] = [x["macro_f1"] for x in tl]
            out[f"{proto}|{feat}|{m}"] = row
            L.append(f"| {MODEL_NAME[m]} | {ms(row['accuracy'])} | {ms(row['macro_f1'])} | {ms(row['auc'], False)} | "
                     f"{ms(row['worst_env_acc'])} | {ms(row['pred_share_fake'])} | {row['collapsed']}/{len(rs)} | "
                     f"{ms(row['last_macro_f1'])} |")
        L.append("")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
