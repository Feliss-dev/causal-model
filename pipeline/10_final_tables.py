"""
12_final_tables.py
==================
Gộp toàn bộ kết quả (cũ + mới) thành bảng cuối cùng cho paper, mean±std 3 seeds.

Nguồn:
  - CausalHeteroGNN + BaselineBranch : metrics_main_s* / metrics_bd_s*
  - IRM / EERM                       : baselines_irm_eerm_{stdood,conf}_s*
  - ERM / MLP (mới)                  : baselines_erm_mlp_{stdood,conf}_s*
  - Domain ablation (mới)            : metrics_nd_s*
  - Worst-group                      : worst_group_{stdood,conf}.json

Output: results/final_tables.md + results/final_tables.json
"""
import os
import json
import glob
import numpy as np

R = "results"


def agg(paths, *keys):
    """mean±std của 1 field qua các seed file."""
    vals = []
    for p in paths:
        d = json.load(open(p))
        for k in keys:
            d = d[k]
        vals.append(float(d))
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def row(name, paths, prefix=None):
    """Trả về dict các metric chuẩn cho 1 model. prefix = key gốc trong json."""
    def k(*keys):
        return agg(paths, *( [prefix] if prefix else [] ), *keys)
    seen_acc = k("seen", "accuracy")
    ood_acc  = k("unseen", "accuracy")
    ood_f1   = k("unseen", "f1")
    ood_auc  = k("unseen", "auc")
    f1_drop  = k("f1_drop_pct")
    return {
        "model": name, "n_seeds": seen_acc[2],
        "seen_acc": seen_acc[:2], "ood_acc": ood_acc[:2],
        "ood_f1": ood_f1[:2], "ood_auc": ood_auc[:2], "f1_drop": f1_drop[:2],
    }


def fmt(r):
    def pm(v, scale=1.0, digits=1):
        return f"{v[0]*scale:.{digits}f}±{v[1]*scale:.{digits}f}"
    return (f"| {r['model']:<22} | {pm(r['seen_acc'],100)} | {pm(r['ood_acc'],100)} | "
            f"{pm(r['ood_f1'],1,3)} | {pm(r['ood_auc'],1,3)} | {pm(r['f1_drop'],1,1)} |")


def table(title, rows):
    lines = [f"\n## {title}\n",
             "| Model | Seen Acc% | OOD Acc% | OOD F1 | OOD AUC | F1 Drop% |",
             "|---|---|---|---|---|---|"]
    lines += [fmt(r) for r in rows]
    return "\n".join(lines)


def main():
    out_rows = {}

    # ---------- Standard OOD ----------
    std = []
    main_files = sorted(glob.glob(f"{R}/metrics_main_s*.json"))
    erm_std    = sorted(glob.glob(f"{R}/baselines_erm_mlp_stdood_s*.json"))
    irm_std    = sorted(glob.glob(f"{R}/baselines_irm_eerm_stdood_s*.json"))
    std.append(row("MLP (content-only)", erm_std, "mlp"))
    std.append(row("ERM HeteroSAGE", erm_std, "erm"))
    std.append(row("IRM", irm_std, "irm"))
    std.append(row("EERM", irm_std, "eerm"))
    std.append(row("BaselineBranch*", main_files, "baseline"))
    std.append(row("CausalHeteroGNN", main_files, "causal"))
    out_rows["standard_ood"] = std

    # ---------- Confounding-shift ----------
    conf = []
    bd_files  = sorted(glob.glob(f"{R}/metrics_bd_s*.json"))
    erm_conf  = sorted(glob.glob(f"{R}/baselines_erm_mlp_conf_s*.json"))
    irm_conf  = sorted(glob.glob(f"{R}/baselines_irm_eerm_conf_s*.json"))
    conf.append(row("MLP (content-only)", erm_conf, "mlp"))
    conf.append(row("ERM HeteroSAGE", erm_conf, "erm"))
    conf.append(row("IRM", irm_conf, "irm"))
    conf.append(row("EERM", irm_conf, "eerm"))
    conf.append(row("BaselineBranch*", bd_files, "baseline"))
    conf.append(row("CausalHeteroGNN", bd_files, "causal"))
    out_rows["confounding_shift"] = conf

    # ---------- Domain ablation ----------
    nd_files = sorted(glob.glob(f"{R}/metrics_nd_s*.json"))
    abl = []
    if nd_files:
        abl.append(row("Causal (full)", bd_files, "causal"))
        abl.append(row("Causal (domain=0.5)", nd_files, "causal"))
        out_rows["domain_ablation"] = abl

    # ---------- Improvement variants (CLIPcons, 5 seeds nếu có) ----------
    cc_main = sorted(glob.glob(f"{R}/metrics_ccmain_s*.json"))
    cc_bd   = sorted(glob.glob(f"{R}/metrics_ccbd_s*.json"))
    cc_std7 = sorted(glob.glob(f"{R}/baselines_erm_mlp_ccstd_s*.json"))
    cc_cf7  = sorted(glob.glob(f"{R}/baselines_erm_mlp_ccconf_s*.json"))
    imp_std, imp_conf = [], []
    if cc_main:
        imp_std.append(row("Causal (goc, 3 seeds)", main_files, "causal"))
        imp_std.append(row("Causal + CLIPcons", cc_main, "causal"))
        imp_std.append(row("MLP + CLIPcons", cc_std7, "mlp"))
        imp_std.append(row("ERM + CLIPcons", cc_std7, "erm"))
        out_rows["improved_standard"] = imp_std
    if cc_bd:
        imp_conf.append(row("Causal (goc, 3 seeds)", bd_files, "causal"))
        imp_conf.append(row("Causal + CLIPcons", cc_bd, "causal"))
        imp_conf.append(row("MLP + CLIPcons", cc_cf7, "mlp"))
        imp_conf.append(row("ERM + CLIPcons", cc_cf7, "erm"))
        out_rows["improved_conf"] = imp_conf

    # ---------- AutoCut v2 discovery summary ----------
    ac_lines = []
    for proto, pat in [("confounding-shift", f"{R}/autocut_search_conf_s*.json"),
                       ("standard", f"{R}/autocut_search_std_s*.json")]:
        for p in sorted(glob.glob(pat)):
            d = json.load(open(p))
            seed = d["_config"]["seed"]
            ch = d["chosen_cut"]
            cand = {c["cut"]: c for c in d["candidates"]}
            probe = cand[ch]["env_probe_acc_val"]
            oacc = cand[ch]["ood_acc_REPORT_ONLY"]
            ac_lines.append(f"| {proto} | s{seed} | **{ch}** | {probe:.3f} | {oacc*100:.1f} |")

    md = ["# Final Result Tables (mean±std, 3 seeds)\n",
          "`BaselineBranch*` = nhánh baseline bên trong CausalHeteroGNN "
          "(chung encoder, KHÔNG phải baseline độc lập — chỉ để tham khảo; "
          "baseline độc lập đúng nghĩa là `ERM HeteroSAGE`).",
          table("Standard OOD (held-out r/neutralnews + r/theonion, "
                "inductive content-only)", std),
          table("Confounding-Shift (ρ=0.9 → 0.1, transductive)", conf)]
    if abl:
        md.append(table("Domain label-history ablation (confounding-shift)", abl))
    if imp_std:
        md.append(table("Improvement v2 — Standard OOD (CLIPcons = "
                        "cosine(CLIP-text, CLIP-image), 5 seeds)", imp_std))
    if imp_conf:
        md.append(table("Improvement v2 — Confounding-Shift (CLIPcons, 5 seeds)", imp_conf))
    if ac_lines:
        md.append("\n## AutoCut v2 — tự khám phá confounder "
                  "(chọn bằng env-probe trên val, KHÔNG nhìn OOD)\n")
        md.append("| Protocol | Seed | Cut được chọn | env-probe | OOD acc% (đối chiếu) |")
        md.append("|---|---|---|---|---|")
        md.extend(ac_lines)
        md.append("\nConfounded: 3/3 seeds chọn đúng `posted_in+member_of` (= cô lập "
                  "Subreddit). Standard: selection bất ổn định (không có confounder "
                  "trội) — đúng hành vi mong muốn của tiêu chí.")

    # ---------- Worst-group ----------
    for proto in ["stdood", "conf"]:
        p = f"{R}/worst_group_{proto}.json"
        if not os.path.exists(p):
            continue
        wg = json.load(open(p))["summary"]
        md.append(f"\n## Worst-group accuracy — {proto}\n")
        md.append("| Model | Worst-Group Acc | Avg-Group Acc |")
        md.append("|---|---|---|")
        for name, s in wg.items():
            md.append(f"| {name} | {s['worst_group_acc'][0]*100:.1f}±"
                      f"{s['worst_group_acc'][1]*100:.1f} | "
                      f"{s['avg_group_acc'][0]*100:.1f}±"
                      f"{s['avg_group_acc'][1]*100:.1f} |")

    text = "\n".join(md) + "\n"
    with open(f"{R}/final_tables.md", "w", encoding="utf-8") as f:
        f.write(text)
    with open(f"{R}/final_tables.json", "w") as f:
        json.dump(out_rows, f, indent=4)
    print(text)
    print(f"\nĐã lưu {R}/final_tables.md + final_tables.json")


if __name__ == "__main__":
    main()
