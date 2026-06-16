"""
aggregate_loco.py - Tong hop ket qua LOCO 3 seeds, 5 mo hinh.
"""
import json, statistics, os

seeds  = [42, 1, 2]
folds  = [
    ("locoa", "nottheonion + pareidolia       "),
    ("locob", "upliftingnews + fakehistoryporn"),
    ("lococ", "usnews+usanews+fakealbumcovers "),
]

def load(path, *keys):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    for k in keys:
        d = d.get(k, {})
    return d if isinstance(d, float) else d.get("accuracy") if isinstance(d, dict) else None

models = {
    "CausalHeteroGNN": lambda f, s: load(f"results/metrics_{f}_s{s}.json",   "causal", "unseen", "accuracy"),
    "Baseline GNN":    lambda f, s: load(f"results/baselines_erm_mlp_{f}_s{s}.json", "erm", "unseen", "accuracy"),
    "MLP":             lambda f, s: load(f"results/baselines_erm_mlp_{f}_s{s}.json", "mlp", "unseen", "accuracy"),
    "IRM":             lambda f, s: load(f"results/baselines_irm_eerm_{f}_s{s}.json","irm", "unseen", "accuracy"),
    "EERM":            lambda f, s: load(f"results/baselines_irm_eerm_{f}_s{s}.json","eerm","unseen", "accuracy"),
}

# Fix load function for nested dicts
def get_acc(path, *keys):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return None
    if isinstance(d, float):
        return d
    if isinstance(d, dict):
        return d.get("accuracy")
    return None

models = {
    "CausalHeteroGNN": lambda f,s: get_acc(f"results/metrics_{f}_s{s}.json",
                                            "causal","unseen","accuracy"),
    "Baseline GNN":    lambda f,s: get_acc(f"results/baselines_erm_mlp_{f}_s{s}.json",
                                            "erm","unseen","accuracy"),
    "MLP":             lambda f,s: get_acc(f"results/baselines_erm_mlp_{f}_s{s}.json",
                                            "mlp","unseen","accuracy"),
    "IRM":             lambda f,s: get_acc(f"results/baselines_irm_eerm_{f}_s{s}.json",
                                            "irm","unseen","accuracy"),
    "EERM":            lambda f,s: get_acc(f"results/baselines_irm_eerm_{f}_s{s}.json",
                                            "eerm","unseen","accuracy"),
}

print("=" * 100)
print("LOCO RESULTS — 5 models x 3 folds x 3 seeds (Accuracy % on held-out fold)")
print("=" * 100)

# Per-model table
agg = {}   # model -> fold -> [acc_s42, acc_s1, acc_s2]

for mname, loader in models.items():
    print(f"\n{'─'*90}")
    print(f"  {mname}")
    print(f"{'─'*90}")
    print(f"  {'Fold':<35} {'s=42':>7} {'s=1':>7} {'s=2':>7} {'mean':>8} {'std':>7}")
    print(f"  {'-'*75}")
    fold_means = []
    agg[mname] = {}
    for ftag, flabel in folds:
        vals = []
        for s in seeds:
            v = loader(ftag, s)
            vals.append(v)
        valid = [v for v in vals if v is not None]
        m = statistics.mean(valid) * 100 if valid else float("nan")
        sd = statistics.stdev(valid) * 100 if len(valid) >= 2 else 0.0
        fold_means.append(m)
        agg[mname][ftag] = {"vals": vals, "mean": m, "std": sd}
        vstr = [f"{v*100:7.2f}" if v else "   N/A " for v in vals]
        print(f"  {flabel} {'  '.join(vstr)}  {m:8.2f}  {sd:7.2f}")

    gm  = statistics.mean(fold_means)
    gsd = statistics.stdev(fold_means)
    print(f"  {'Grand mean (across folds)':<35} {'':>7} {'':>7} {'':>7}  {gm:8.2f}  {gsd:7.2f}")

# ── Summary table ────────────────────────────────────────────────────────────
print("\n\n" + "=" * 100)
print("SUMMARY TABLE — mean+/-std across 3 seeds per fold (Accuracy %)")
print("=" * 100)

col_w = 16
header = f"{'Fold':<34}" + "".join(f"{m:>{col_w}}" for m in models)
print(header)
print("-" * (34 + col_w * len(models)))

fold_grand_means = {m: [] for m in models}

for ftag, flabel in folds:
    row = f"{flabel.strip():<34}"
    for mname in models:
        a = agg[mname][ftag]
        m, s = a["mean"], a["std"]
        fold_grand_means[mname].append(m)
        cell = f"{m:.1f}+/-{s:.1f}"
        row += f"{cell:>{col_w}}"
    print(row)

# Grand mean row
print("-" * (34 + col_w * len(models)))
row = f"{'Grand mean (folds)':<34}"
for mname in models:
    vals = fold_grand_means[mname]
    gm  = statistics.mean(vals)
    gsd = statistics.stdev(vals)
    cell = f"{gm:.1f}+/-{gsd:.1f}"
    row += f"{cell:>{col_w}}"
print(row)

# ── Markdown table for paper ─────────────────────────────────────────────────
print("\n\n=== MARKDOWN (copy to paper) ===")
mnames = list(models.keys())
header_md = "| Fold held-out | " + " | ".join(mnames) + " |"
sep_md    = "|" + "|".join(["---"] + ["---:"] * len(mnames)) + "|"
print(header_md)
print(sep_md)

for ftag, flabel in folds:
    cells = []
    for mname in mnames:
        a = agg[mname][ftag]
        cells.append(f"{a['mean']:.1f}+/-{a['std']:.1f}")
    print(f"| {flabel.strip()} | " + " | ".join(cells) + " |")

grand_cells = []
for mname in mnames:
    vals = fold_grand_means[mname]
    gm  = statistics.mean(vals)
    gsd = statistics.stdev(vals)
    grand_cells.append(f"**{gm:.1f}+/-{gsd:.1f}**")
print(f"| **Mean+/-std** | " + " | ".join(grand_cells) + " |")
