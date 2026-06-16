"""
11_generate_figures.py
======================
Gọi tuần tự 5 script trong thư mục files/ để sinh tất cả hình minh họa.

Yêu cầu trước khi chạy:
  - results/final_tables.json      (từ 10_final_tables.py)
  - results/worst_group_conf.json  (từ 09_worst_group.py với confounding-shift)
  - results/metrics_bd_s42.json    (từ 06_evaluate.py, seed 42, SKIP_EXPLAIN=0)

Output: figures/image1.png  image2.png  image3.png  image5.png  image6.png

Chạy:  uv run python pipeline/11_generate_figures.py
"""
import subprocess
import sys
import os

FIGURE_SCRIPTS = [
    "files/fig1_neo4j_schema.py",
    "files/fig2_architecture.py",
    "files/fig3_ood_performance.py",
    "files/fig4_robustness.py",
    "files/fig5_dashboard.py",
]

EXPECTED_OUTPUTS = [
    "figures/image1.png",
    "figures/image2.png",
    "figures/image3.png",
    "figures/image5.png",
    "figures/image6.png",
]


def main():
    for script in FIGURE_SCRIPTS:
        print(f"\n>>> python {script}")
        result = subprocess.run([sys.executable, script], check=True)

    print("\n--- Kiểm tra output ---")
    all_ok = True
    for path in EXPECTED_OUTPUTS:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  OK  {path}  ({size_kb:.0f} KB)")
        else:
            print(f"  MISSING  {path}")
            all_ok = False

    if all_ok:
        print("\nTất cả 5 hình đã được sinh thành công.")
    else:
        print("\nCó hình chưa được sinh — kiểm tra lỗi phía trên.")
        sys.exit(1)


if __name__ == "__main__":
    main()
