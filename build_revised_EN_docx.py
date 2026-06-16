"""
build_revised_EN_docx.py
========================
Converts Fair_Article/paper_revised_EN.md → Fair_Article/paper_revised_EN.docx
using pypandoc (pandoc 3.x). Embeds all figures from Fair_Article/figures/.

Run: PYTHONUTF8=1 uv run python build_revised_EN_docx.py
  or: .\.venv\Scripts\python.exe build_revised_EN_docx.py
"""
import os
import sys

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
SRC    = os.path.join(BASE, "Fair_Article", "paper_revised_EN.md")
DST    = os.path.join(BASE, "Fair_Article", "paper_revised_EN.docx")
RESDIR = os.path.join(BASE, "Fair_Article")   # figures/ lives here

# ── sanity: check source exists ───────────────────────────────────────────────
if not os.path.isfile(SRC):
    sys.exit(f"ERROR: source not found → {SRC}")

# ── check all figures exist ───────────────────────────────────────────────────
EXPECTED_FIGS = [
    "fig_neo4j_schema.png",
    "fig_graph_stats.png",
    "fig2_standard_vs_shift.png",
    "fig1_confounding_ood_bar.png",
    "fig3_confounding_metrics.png",
    "fig7_confusion.png",
    "fig6_fastrp_leak.png",
    "fig5_lfr.png",
    "fig4_f1_drop.png",
    "fig_bi_overview.png",
]
missing = []
for f in EXPECTED_FIGS:
    p = os.path.join(RESDIR, "figures", f)
    if not os.path.isfile(p):
        missing.append(p)

if missing:
    print("WARNING: missing figures:")
    for m in missing:
        print(f"  {m}")
else:
    print(f"[OK] All {len(EXPECTED_FIGS)} figures found in {RESDIR}/figures/")

# ── content validation ────────────────────────────────────────────────────────
with open(SRC, encoding="utf-8") as fh:
    md = fh.read()

checks = {
    "Title":        "Causal Graph Disentanglement via Heterogeneous GraphSAGE",
    "Abstract":     "## Abstract",
    "Sec 1":        "## 1. Introduction",
    "Sec 4":        "## 4. Methodology",
    "Sec 6":        "## 6. Results",
    "Table 6.2":    "Table 6.2",
    "Key number":   "74.2%",
    "LFR=0":        "LFR = 0.0%",
    "AUC 0.851":    "0.851",
    "Fig 1a ref":   "fig_neo4j_schema.png",
    "Fig 9 ref":    "fig_bi_overview.png",
    "Ref [14]":     "[14] Z. Liu",
}
failed = []
for label, token in checks.items():
    if token not in md:
        failed.append(f"  MISSING [{label}]: '{token}'")

if failed:
    print("VALIDATION FAILURES:")
    for f in failed:
        print(f)
    sys.exit("Aborting — fix markdown before converting.")
else:
    print(f"[OK] All {len(checks)} content checks passed.")

# ── pandoc conversion ─────────────────────────────────────────────────────────
import pypandoc  # noqa: E402

extra_args = [
    "--standalone",
    "--toc",
    "--toc-depth=3",
    f"--resource-path={RESDIR}",   # resolves figures/fig_*.png relative paths
    "--wrap=preserve",
    "--reference-links",
    # Page geometry (A4, narrow margins for tables)
    "-V", "geometry:a4paper,margin=2.5cm",
]

print(f"\nConverting:\n  {SRC}\n  → {DST}")
print(f"  resource-path: {RESDIR}")
print(f"  pandoc version: {pypandoc.get_pandoc_version()}")

pypandoc.convert_file(
    SRC,
    "docx",
    outputfile=DST,
    format="markdown+tex_math_dollars+pipe_tables+implicit_figures",
    extra_args=extra_args,
)

size_kb = os.path.getsize(DST) / 1024
print(f"\n[DONE]  {DST}  ({size_kb:.0f} KB)")

# ── verify figure count in docx ───────────────────────────────────────────────
try:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(DST)
    img_count = sum(
        1
        for para in doc.paragraphs
        for run in para.runs
        if run._r.findall(f".//{qn('a:blip')}") or run._r.findall(f".//{qn('pic:pic')}")
    )
    # Also count inline images via relationship check
    img_rels = [
        r for r in doc.part.rels.values()
        if "image" in r.reltype
    ]
    print(f"[VERIFY] {len(img_rels)} image relationship(s) embedded in docx "
          f"(expected {len(EXPECTED_FIGS)}).")
    if len(img_rels) < len(EXPECTED_FIGS):
        print("  WARNING: fewer images embedded than expected — check figure paths in markdown.")
except Exception as e:
    print(f"[VERIFY] Could not inspect docx: {e}")
