"""
export_docx.py
==============
Chuyển 2 bản markdown (EN + VN) sang .docx bằng pandoc (đính kèm qua pypandoc-binary).
Giữ bảng, hình (figures/*.png), và công thức toán ($...$ -> Word equations), kèm mục lục.

Chạy: PYTHONUTF8=1 uv run python export_docx.py
"""
import os
import pypandoc

JOBS = [
    ("Fair_Article/paper_full.md",       "Fair_Article/paper_full.docx"),
    ("Fair_Article_VN/paper_full_VN.md", "Fair_Article_VN/paper_full_VN.docx"),
]

for src, dst in JOBS:
    resource_dir = os.path.dirname(os.path.abspath(src))
    extra_args = [
        "--standalone",
        "--toc",                      # mục lục
        "--toc-depth=2",
        f"--resource-path={resource_dir}",  # để figures/*.png phân giải đúng
        "--wrap=preserve",
    ]
    pypandoc.convert_file(
        src, "docx", outputfile=dst,
        format="markdown+tex_math_dollars+pipe_tables",
        extra_args=extra_args,
    )
    size = os.path.getsize(dst)
    print(f"OK  {src}  ->  {dst}  ({size/1024:.0f} KB)")

print("Done.")
