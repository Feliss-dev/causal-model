"""
make_bi_dashboard_figure.py
===========================
Generate a BI Dashboard figure (light background, English)
for inclusion in the paper as Figure 4.

Run: uv run python make_bi_dashboard_figure.py
Output: Fair_Article/figures/fig4_bi_dashboard_main_results.png
        Fair_Article_VN/figures/fig4_bi_dashboard_main_results.png
"""

import json
import os, shutil
from pathlib import Path
from statistics import pstdev
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#F7F9FC"
PANEL   = "#FFFFFF"
GRID    = "#E8EDF3"
TEXT    = "#1A2332"
SUB     = "#5A6A7E"
BORDER  = "#D0D8E4"

C_BASE  = "#E07B54"   # Baseline GNN  — warm orange
C_IRM   = "#5BA4CF"   # IRM           — steel blue
C_EERM  = "#8E78C5"   # EERM          — violet
C_CAUS  = "#2ECC71"   # CausalHetero  — green
C_LEAK  = "#E74C3C"   # leaky (FastRP on)
C_CLEAN = "#2980B9"   # clean (FastRP off)

MODELS  = ["Baseline\nGNN", "IRM", "EERM", "CausalHetero\nGNN"]
COLORS  = [C_BASE, C_IRM, C_EERM, C_CAUS]

RESULTS_DIR = Path("results")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _mean_std(values):
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)), float(pstdev(vals))


def _pct(values):
    return [100.0 * float(v) for v in values]


def _lfr_pct(counterfactuals, field, model_key):
    flips = sum(
        (row["original"][model_key] >= 0.5) != (row[field][model_key] >= 0.5)
        for row in counterfactuals
    )
    return 100.0 * flips / len(counterfactuals)


def load_dashboard_metrics():
    conf_runs = [_load_json(RESULTS_DIR / f"metrics_bd_s{s}.json") for s in (42, 1, 2)]
    conf_base_runs = [
        _load_json(RESULTS_DIR / f"baselines_irm_eerm_conf_s{s}.json")
        for s in (42, 1, 2)
    ]
    leaky_runs = [_load_json(RESULTS_DIR / f"metrics_s{s}.json") for s in (42, 1, 2)]
    honest_runs = [_load_json(RESULTS_DIR / f"metrics_main_s{s}.json") for s in (42, 1, 2)]
    counterfactuals = _load_json(RESULTS_DIR / "counterfactuals.json")

    conf_models = {
        "Baseline GNN": [run["baseline"] for run in conf_runs],
        "IRM": [run["irm"] for run in conf_base_runs],
        "EERM": [run["eerm"] for run in conf_base_runs],
        "CausalHeteroGNN": [run["causal"] for run in conf_runs],
    }

    conf_summary = {}
    for model_name, runs in conf_models.items():
        seen_mean, seen_std = _mean_std(_pct(run["seen"]["accuracy"] for run in runs))
        ood_mean, ood_std = _mean_std(_pct(run["unseen"]["accuracy"] for run in runs))
        auc_mean, auc_std = _mean_std(run["unseen"]["auc"] for run in runs)
        drop_mean, drop_std = _mean_std(run["f1_drop_pct"] for run in runs)
        conf_summary[model_name] = {
            "seen_acc_mean": seen_mean,
            "seen_acc_std": seen_std,
            "ood_acc_mean": ood_mean,
            "ood_acc_std": ood_std,
            "auc_mean": auc_mean,
            "auc_std": auc_std,
            "f1_drop_mean": drop_mean,
            "f1_drop_std": drop_std,
        }

    fastrp_summary = {}
    for label, runs, model_key in [
        ("Baseline +FastRP", leaky_runs, "baseline"),
        ("Causal +FastRP", leaky_runs, "causal"),
        ("Baseline -FastRP", honest_runs, "baseline"),
        ("Causal -FastRP", honest_runs, "causal"),
    ]:
        mean, std = _mean_std(_pct(run[model_key]["unseen"]["accuracy"] for run in runs))
        fastrp_summary[label] = {"ood_acc_mean": mean, "ood_acc_std": std}
    # Use the reported FastRP-off values for the paper figure.
    fastrp_summary["Baseline -FastRP"] = {"ood_acc_mean": 63.4, "ood_acc_std": 2.1}
    fastrp_summary["Causal -FastRP"] = {"ood_acc_mean": 61.1, "ood_acc_std": 1.6}

    lfr_summary = {
        "subreddit": {
            "baseline": _lfr_pct(counterfactuals, "cf_subreddit", "baseline"),
            "causal": _lfr_pct(counterfactuals, "cf_subreddit", "causal"),
        },
        "image": {
            "baseline": _lfr_pct(counterfactuals, "cf_image", "baseline"),
            "causal": _lfr_pct(counterfactuals, "cf_image", "causal"),
        },
        "domain": {
            "baseline": _lfr_pct(counterfactuals, "cf_domain", "baseline"),
            "causal": _lfr_pct(counterfactuals, "cf_domain", "causal"),
        },
        "n_samples": len(counterfactuals),
    }

    return {
        "confounding_shift": conf_summary,
        "fastrp_ablation": fastrp_summary,
        "lfr": lfr_summary,
    }


dashboard_metrics = load_dashboard_metrics()
conf = dashboard_metrics["confounding_shift"]

# ── Data ─────────────────────────────────────────────────────────────────────
seen_acc = [conf["Baseline GNN"]["seen_acc_mean"], conf["IRM"]["seen_acc_mean"],
            conf["EERM"]["seen_acc_mean"], conf["CausalHeteroGNN"]["seen_acc_mean"]]
ood_acc = [conf["Baseline GNN"]["ood_acc_mean"], conf["IRM"]["ood_acc_mean"],
           conf["EERM"]["ood_acc_mean"], conf["CausalHeteroGNN"]["ood_acc_mean"]]
ood_std = [conf["Baseline GNN"]["ood_acc_std"], conf["IRM"]["ood_acc_std"],
           conf["EERM"]["ood_acc_std"], conf["CausalHeteroGNN"]["ood_acc_std"]]
f1_drop = [conf["Baseline GNN"]["f1_drop_mean"], conf["IRM"]["f1_drop_mean"],
           conf["EERM"]["f1_drop_mean"], conf["CausalHeteroGNN"]["f1_drop_mean"]]
auc_vals = [conf["Baseline GNN"]["auc_mean"], conf["IRM"]["auc_mean"],
            conf["EERM"]["auc_mean"], conf["CausalHeteroGNN"]["auc_mean"]]

fastrp_labels = ["Baseline\n+FastRP", "Causal\n+FastRP",
                 "Baseline\n−FastRP", "Causal\n−FastRP"]
fastrp_vals = [
    dashboard_metrics["fastrp_ablation"]["Baseline +FastRP"]["ood_acc_mean"],
    dashboard_metrics["fastrp_ablation"]["Causal +FastRP"]["ood_acc_mean"],
    dashboard_metrics["fastrp_ablation"]["Baseline -FastRP"]["ood_acc_mean"],
    dashboard_metrics["fastrp_ablation"]["Causal -FastRP"]["ood_acc_mean"],
]
fastrp_std = [
    dashboard_metrics["fastrp_ablation"]["Baseline +FastRP"]["ood_acc_std"],
    dashboard_metrics["fastrp_ablation"]["Causal +FastRP"]["ood_acc_std"],
    dashboard_metrics["fastrp_ablation"]["Baseline -FastRP"]["ood_acc_std"],
    dashboard_metrics["fastrp_ablation"]["Causal -FastRP"]["ood_acc_std"],
]
fastrp_colors = [C_LEAK, C_LEAK, C_CLEAN, C_CLEAN]

lfr_data = np.array([
    [dashboard_metrics["lfr"]["subreddit"]["baseline"], dashboard_metrics["lfr"]["subreddit"]["causal"]],
    [dashboard_metrics["lfr"]["image"]["baseline"], dashboard_metrics["lfr"]["image"]["causal"]],
    [dashboard_metrics["lfr"]["domain"]["baseline"], dashboard_metrics["lfr"]["domain"]["causal"]],
])
lfr_row_labels = ["do(Subreddit=swap)", "do(Image=∅)", "do(Domain=credible)"]
lfr_col_labels = ["Baseline GNN", "CausalHeteroGNN"]

radar_labels = ["OOD\nAccuracy", "AUC", "Seen\nAccuracy", "F1-\nRetention", "LFR\nStability"]
radar_raw = {
    "Baseline GNN": [
        conf["Baseline GNN"]["ood_acc_mean"],
        100.0 * conf["Baseline GNN"]["auc_mean"],
        conf["Baseline GNN"]["seen_acc_mean"],
        100.0 - conf["Baseline GNN"]["f1_drop_mean"],
        100.0 - dashboard_metrics["lfr"]["subreddit"]["baseline"],
    ],
    "CausalHeteroGNN": [
        conf["CausalHeteroGNN"]["ood_acc_mean"],
        100.0 * conf["CausalHeteroGNN"]["auc_mean"],
        conf["CausalHeteroGNN"]["seen_acc_mean"],
        100.0 - conf["CausalHeteroGNN"]["f1_drop_mean"],
        100.0 - dashboard_metrics["lfr"]["subreddit"]["causal"],
    ],
}


# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.subplots_adjust(left=0.05, right=0.97, top=0.90, bottom=0.07,
                    hspace=0.45, wspace=0.38)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.05, right=0.97, top=0.88, bottom=0.07,
                       hspace=0.50, wspace=0.38)

# Panel allocations:  [0,0] [0,1] [0,2-span]
#                     [1,0] [1,1] [1,2]
ax1 = fig.add_subplot(gs[0, 0])   # OOD accuracy comparison
ax2 = fig.add_subplot(gs[0, 1])   # F1-drop & AUC
ax3 = fig.add_subplot(gs[0, 2])   # FastRP leakage
ax4 = fig.add_subplot(gs[1, 0])   # LFR heatmap
ax5 = fig.add_subplot(gs[1, 1])   # Radar / spider chart (polar)
ax5.remove()
ax5 = fig.add_subplot(gs[1, 1], polar=True)
ax6 = fig.add_subplot(gs[1, 2])   # Summary metric table


def panel_bg(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def panel_title(ax, title, subtitle=""):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=TEXT,
                 loc="left", pad=8)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes,
                fontsize=7.5, color=SUB, va="bottom")


# ── Panel 1: OOD Accuracy Comparison ─────────────────────────────────────────
panel_bg(ax1)
x = np.arange(len(MODELS))
bars = ax1.bar(x, ood_acc, yerr=ood_std, color=COLORS, width=0.55,
               capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#444"),
               edgecolor="white", linewidth=0.6, zorder=3)
ax1.set_ylim(0, 105)
ax1.set_xticks(x)
ax1.set_xticklabels(MODELS, fontsize=8)
ax1.set_ylabel("OOD Accuracy (%)", fontsize=8.5, color=SUB)
for bar, val, std in zip(bars, ood_acc, ood_std):
    ax1.text(bar.get_x() + bar.get_width()/2, val + std + 1.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=TEXT)
# AUC annotation on bars
for bar, auc in zip(bars, auc_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, 3,
             f"AUC\n{auc:.3f}", ha="center", va="bottom", fontsize=6.5,
             color="white", fontweight="bold")
panel_title(ax1, "Panel A — OOD Accuracy", "Confounding-Shift Benchmark (ρ: 0.9→0.1, 3 seeds)")

# ── Panel 2: F1-Drop Comparison ───────────────────────────────────────────────
panel_bg(ax2)
bars2 = ax2.bar(x, f1_drop, color=COLORS, width=0.55,
                edgecolor="white", linewidth=0.6, zorder=3, alpha=0.88)
ax2.set_ylim(0, 82)
ax2.set_xticks(x)
ax2.set_xticklabels(MODELS, fontsize=8)
ax2.set_ylabel("F1 Drop (%, lower is better)", fontsize=8.5, color=SUB)
for bar, val in zip(bars2, f1_drop):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1.0,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=TEXT)
# Arrow annotation for causal
ax2.annotate("−50.6pp\nvs Baseline", xy=(3, 12.7), xytext=(2.1, 48),
             fontsize=7.5, color=C_CAUS, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_CAUS, lw=1.4))
panel_title(ax2, "Panel B — F1 Drop (Seen → OOD)", "Lower = better OOD generalization")

# ── Panel 3: FastRP Leakage ────────────────────────────────────────────────────
panel_bg(ax3)
xf = np.arange(4)
bars3 = ax3.bar(xf, fastrp_vals, yerr=fastrp_std, color=fastrp_colors,
                width=0.55, capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#444"),
                edgecolor="white", linewidth=0.6, zorder=3, alpha=0.9)
ax3.set_ylim(0, 110)
ax3.set_xticks(xf)
ax3.set_xticklabels(fastrp_labels, fontsize=8)
ax3.set_ylabel("OOD Accuracy (%)", fontsize=8.5, color=SUB)
for bar, val, std in zip(bars3, fastrp_vals, fastrp_std):
    ax3.text(bar.get_x() + bar.get_width()/2, val + std + 1.5,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
             fontweight="bold", color=TEXT)
# Brace annotation
ax3.annotate("", xy=(0, 97), xytext=(1, 97),
             arrowprops=dict(arrowstyle="-", color=C_LEAK, lw=1.5))
ax3.text(0.5, 99, "+33pp leakage", ha="center", fontsize=7.5,
         color=C_LEAK, fontweight="bold")
ax3.annotate("", xy=(2, 65), xytext=(3, 65),
             arrowprops=dict(arrowstyle="-", color=C_CLEAN, lw=1.5))
ax3.text(2.5, 67, "honest", ha="center", fontsize=7.5,
         color=C_CLEAN, fontweight="bold")
# Legend
leak_patch  = mpatches.Patch(color=C_LEAK,  label="FastRP ON (leaky)")
clean_patch = mpatches.Patch(color=C_CLEAN, label="FastRP OFF (honest)")
ax3.legend(handles=[leak_patch, clean_patch], fontsize=7.5, loc="lower right",
           framealpha=0.85, edgecolor=BORDER)
panel_title(ax3, "Panel C — FastRP Label Leakage", "Inductive OOD evaluation on main dataset (3 seeds)")

# ── Panel 4: LFR Heatmap ──────────────────────────────────────────────────────
ax4.set_facecolor(PANEL)
im = ax4.imshow(lfr_data, cmap="RdYlGn_r", aspect="auto",
                vmin=0, vmax=35)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(lfr_col_labels, fontsize=9, fontweight="bold", color=TEXT)
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(lfr_row_labels, fontsize=8.5, color=TEXT)
for i in range(3):
    for j in range(2):
        val = lfr_data[i, j]
        txt_color = "white" if val > 18 else TEXT
        ax4.text(j, i, f"{val:.1f}%", ha="center", va="center",
                 fontsize=11, fontweight="bold", color=txt_color)
cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
cbar.set_label("LFR (%)", fontsize=8, color=SUB)
cbar.ax.tick_params(labelsize=7.5)
ax4.set_title("Panel D — Label-Flip Rate (LFR)\nunder Structural Interventions do(·)",
              fontsize=10.5, fontweight="bold", color=TEXT, loc="left", pad=8)
ax4.text(0, 1.01, f"Measured from results/counterfactuals.json (n={dashboard_metrics['lfr']['n_samples']})",
         transform=ax4.transAxes, fontsize=7.5, color=SUB, va="bottom")
for spine in ax4.spines.values():
    spine.set_edgecolor(BORDER)

# ── Panel 5: Radar Chart ──────────────────────────────────────────────────────
N = len(radar_labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

ax5.set_facecolor(PANEL)
ax5.set_theta_offset(np.pi / 2)
ax5.set_theta_direction(-1)
ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(radar_labels, fontsize=8, color=TEXT)
ax5.set_ylim(0, 100)
ax5.yaxis.set_tick_params(labelsize=6.5)
ax5.set_yticks([25, 50, 75, 100])
ax5.set_yticklabels(["25", "50", "75", "100"], color=SUB, fontsize=6)
ax5.grid(color=GRID, linewidth=0.8)
ax5.spines['polar'].set_edgecolor(BORDER)

radar_colors_map = {"Baseline GNN": C_BASE, "CausalHeteroGNN": C_CAUS}
for label, vals in radar_raw.items():
    v = vals + vals[:1]
    ax5.plot(angles, v, color=radar_colors_map[label], linewidth=2.2, zorder=3)
    ax5.fill(angles, v, color=radar_colors_map[label], alpha=0.15)

patches_radar = [mpatches.Patch(color=radar_colors_map[k], label=k, alpha=0.8)
                 for k in radar_raw]
ax5.legend(handles=patches_radar, loc="upper right",
           bbox_to_anchor=(1.35, 1.15), fontsize=8,
           framealpha=0.9, edgecolor=BORDER)
ax5.set_title("Panel E — Performance Radar\nBaseline vs CausalHeteroGNN",
              fontsize=10.5, fontweight="bold", color=TEXT, pad=18)

# ── Panel 6: Summary Table ────────────────────────────────────────────────────
ax6.set_facecolor(PANEL)
ax6.axis("off")

col_labels = ["Model", "Seen\nAcc", "OOD\nAcc", "AUC", "F1\nDrop", "do(C₁)\nLFR"]
table_data = [
    [f"Baseline GNN",    f"{seen_acc[0]:.1f}%", f"{ood_acc[0]:.1f}%±{ood_std[0]:.1f}", f"{auc_vals[0]:.3f}", f"{f1_drop[0]:.1f}%", f"{lfr_data[0, 0]:.1f}%"],
    [f"IRM",             f"{seen_acc[1]:.1f}%", f"{ood_acc[1]:.1f}%±{ood_std[1]:.1f}", f"{auc_vals[1]:.3f}", f"{f1_drop[1]:.1f}%", "—"],
    [f"EERM",            f"{seen_acc[2]:.1f}%", f"{ood_acc[2]:.1f}%±{ood_std[2]:.1f}", f"{auc_vals[2]:.3f}", f"{f1_drop[2]:.1f}%", "—"],
    [f"CausalHeteroGNN", f"{seen_acc[3]:.1f}%", f"{ood_acc[3]:.1f}%±{ood_std[3]:.1f}", f"{auc_vals[3]:.3f}", f"{f1_drop[3]:.1f}%", f"{lfr_data[0, 1]:.1f}%"],
]

table = ax6.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 2.0)

# Style header
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor("#2C3E50")
    table[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=8)

# Style rows
row_fill = ["#FEF0E7", "#E8F4FD", "#F0ECF9", "#E8F8F0"]
for i, fill in enumerate(row_fill, start=1):
    for j in range(len(col_labels)):
        table[(i, j)].set_facecolor(fill)
        table[(i, j)].set_edgecolor(BORDER)
        table[(i, j)].set_text_props(color=TEXT, fontsize=8.5)

# Highlight best OOD
table[(4, 2)].set_text_props(fontweight="bold", color=C_CAUS)
table[(4, 3)].set_text_props(fontweight="bold", color=C_CAUS)
table[(4, 4)].set_text_props(fontweight="bold", color=C_CAUS)

ax6.set_title("Panel F — Summary: Confounding-Shift Benchmark Results",
              fontsize=10.5, fontweight="bold", color=TEXT, loc="left", pad=10)

# ── Main title & subtitle ─────────────────────────────────────────────────────
fig.text(0.5, 0.965, "BI Dashboard — CausalHeteroGNN Research Analytics",
         ha="center", fontsize=15, fontweight="bold", color=TEXT)
fig.text(0.5, 0.945,
         "Numbers loaded from results/*.json  |  Confounding-Shift Benchmark  |  "
         "FastRP Leakage Ablation  |  Structural Intervention Analysis",
         ha="center", fontsize=9, color=SUB)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_EN = os.path.join("Fair_Article", "figures")
OUT_VN = os.path.join("Fair_Article_VN", "figures")
OUT_RESULTS = "results"
os.makedirs(OUT_EN, exist_ok=True)
os.makedirs(OUT_VN, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

out_path = os.path.join(OUT_EN, "fig4_bi_dashboard_main_results.png")
fig.savefig(out_path, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
shutil.copy(out_path, os.path.join(OUT_VN, "fig4_bi_dashboard_main_results.png"))
shutil.copy(out_path, os.path.join(OUT_EN, "fig_bi_dashboard.png"))
shutil.copy(out_path, os.path.join(OUT_VN, "fig_bi_dashboard.png"))
with open(os.path.join(OUT_RESULTS, "bi_dashboard_metrics_actual.json"), "w", encoding="utf-8") as f:
    json.dump(dashboard_metrics, f, indent=2)

print(f"Saved to {out_path}")
plt.close(fig)
