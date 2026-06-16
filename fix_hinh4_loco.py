# -*- coding: utf-8 -*-
"""
Regenerate the LOCO figure used as Figure 4 and replace the embedded image in
the polished docx. The chart title no longer contains the stale "Figure 9".
"""

from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np


SRC_DOCX = Path("CausalHeteroGNN_FAIR_VN_Final_trau_chuot.docx")
OUT_DOCX = Path("CausalHeteroGNN_FAIR_VN_Final_trau_chuot_sua_hinh4.docx")
OUT_FIG = Path("figures/fig4_loco_comparison_corrected.png")


def make_corrected_loco_figure():
    labels = [
        "nottheonion\n+pareidolia",
        "upliftingnews\n+fakehistoryporn",
        "usnews+usanews\n+fakealbumcovers",
        "Mean\n(3 folds)",
    ]
    models = [
        "MLP content-only",
        "Baseline GNN",
        "IRM",
        "EERM",
        "CausalHeteroGNN",
    ]
    values = np.array(
        [
            [61.3, 57.9, 62.0, 62.2, 62.3],
            [73.1, 56.8, 66.5, 70.7, 70.5],
            [80.5, 54.7, 69.1, 76.9, 75.5],
            [71.6, 56.5, 65.8, 70.0, 69.4],
        ]
    )
    errors = np.zeros_like(values)
    errors[3] = [7.9, 1.3, 2.9, 6.0, 5.4]

    colors = ["#4c78c7", "#f07f2f", "#70ad47", "#ffc000", "#c00000"]
    x = np.arange(len(labels))
    width = 0.14

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 6.3), dpi=180)

    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            values[:, i],
            width,
            label=model,
            color=color,
            yerr=errors[:, i],
            capsize=4,
            ecolor="#4a4a4a",
            linewidth=0,
        )
        for j, bar in enumerate(bars):
            y = bar.get_height()
            err = errors[j, i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y + err + 0.8,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color=color,
            )

    ax.set_title(
        "LOCO Evaluation on Natural Community Splits\n"
        "5 models compared across 3 held-out folds and mean accuracy",
        fontweight="bold",
        pad=10,
    )
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(38, 96)
    ax.grid(axis="y", alpha=0.28)
    ax.axvline(2.5, color="#bfc5cc", linestyle="--", linewidth=1.5)
    ax.text(2.62, 92.8, "Mean ->", fontsize=10.5, color="#808080", fontstyle="italic")
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)


def replace_docx_image():
    with zipfile.ZipFile(SRC_DOCX, "r") as zin, zipfile.ZipFile(OUT_DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        replacement = OUT_FIG.read_bytes()
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/media/image4.png":
                data = replacement
            zout.writestr(item, data)


def main():
    make_corrected_loco_figure()
    replace_docx_image()
    print(f"Saved corrected figure: {OUT_FIG}")
    print(f"Saved corrected document: {OUT_DOCX}")


if __name__ == "__main__":
    main()
