"""
01_prepare_upfd.py
==================
Chuẩn bị bộ dữ liệu UPFD (User Preference-aware Fake News Detection, Dou et al. SIGIR'21)
cho hướng tiếp cận "đồ thị lan truyền": mỗi tin = một đồ thị (node gốc = tin, các node
còn lại = user retweet), nhãn Fake/Real gắn cho cả đồ thị.

Nguồn: OpenDataLab (opendatalab.com/OpenDataLab/UPFD) không truy cập được từ máy này;
Google Drive của PyG trả 404. Dùng bản sao chính thức của DGL (cùng file raw của tác giả):
  https://data.dgl.ai/dataset/FakeNewsPOL.zip   (PolitiFact, ~198 MB)
  https://data.dgl.ai/dataset/FakeNewsGOS.zip   (GossipCop, ~1.4 GB)
và file thời gian/ID từ repo gốc github.com/safe-graph/GNN-FakeNews (thư mục data/).

Script:
  1. Giải nén zip vào data/upfd/{politifact,gossipcop}/raw (đúng cấu trúc torch_geometric UPFD).
  2. Nạp qua torch_geometric.datasets.UPFD với 4 loại feature (bert, spacy, profile, content)
     để PyG xử lý và cache (giữ nguyên chia train/val/test gốc 20/10/70).
  3. Tính metadata từng đồ thị: nhãn, split, số node/cạnh, độ sâu cây, thời điểm retweet
     đầu tiên (xấp xỉ thời điểm tin xuất hiện) -> môi trường thời gian cho phần nhân quả.
  4. In báo cáo kiểm tra dữ liệu.

Cách chạy (từ project root):
  $env:PYTHONUTF8 = "1"
  uv run python propagation/01_prepare_upfd.py
Output: data/upfd/{name}/graph_meta.csv, results/upfd/data_report.md
"""

import os
import pickle
import zipfile
import numpy as np
import pandas as pd
from collections import deque

ROOT = os.path.join("data", "upfd")
ZIPS = {"politifact": "FakeNewsPOL.zip", "gossipcop": "FakeNewsGOS.zip"}
META = {"politifact": "pol", "gossipcop": "gos"}
FEATURES = ["bert", "spacy", "profile", "content"]
REPORT_DIR = os.path.join("results", "upfd")
os.makedirs(REPORT_DIR, exist_ok=True)


def unzip(name):
    raw = os.path.join(ROOT, name, "raw")
    zpath = os.path.join(ROOT, "_zips", ZIPS[name])
    if os.path.exists(os.path.join(raw, "A.txt")):
        return True
    if not os.path.exists(zpath):
        print(f"[skip] {zpath} chưa có")
        return False
    os.makedirs(raw, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(raw)
    return True


def load_pyg(name):
    from torch_geometric.datasets import UPFD
    sizes = {}
    for f in FEATURES:
        for s in ["train", "val", "test"]:
            ds = UPFD(ROOT, name, f, s)
            sizes[(f, s)] = (len(ds), ds.num_features)
    return sizes


def graph_meta(name):
    raw = os.path.join(ROOT, name, "raw")
    node_graph = np.load(os.path.join(raw, "node_graph_id.npy"))
    labels = np.load(os.path.join(raw, "graph_labels.npy"))
    edges = np.loadtxt(os.path.join(raw, "A.txt"), delimiter=",", dtype=np.int64)
    split = np.empty(len(labels), dtype=object)
    for s in ["train", "val", "test"]:
        split[np.load(os.path.join(raw, f"{s}_idx.npy"))] = s

    p = META[name]
    times = pickle.load(open(os.path.join(ROOT, "meta", f"{p}_id_time_mapping.pkl"), "rb"))
    news_ids = open(os.path.join(ROOT, "meta", f"{p}_news_list.txt")).read().split()

    n_graphs = len(labels)
    t = np.array([float(times[i]) if times.get(i, "") != "" else np.nan
                  for i in range(len(node_graph))])
    df = pd.DataFrame({"graph_id": np.arange(n_graphs), "label": labels, "split": split})
    df["news_id"] = news_ids[:n_graphs] if len(news_ids) >= n_graphs else None
    df["n_nodes"] = np.bincount(node_graph, minlength=n_graphs)
    df["n_edges"] = np.bincount(node_graph[edges[:, 0]], minlength=n_graphs)
    tg = pd.DataFrame({"g": node_graph, "t": t}).groupby("g")["t"]
    df["first_retweet"] = tg.min().reindex(df["graph_id"]).values
    df["last_retweet"] = tg.max().reindex(df["graph_id"]).values
    df["first_retweet_date"] = pd.to_datetime(df["first_retweet"], unit="s").dt.date

    # độ sâu cây lan truyền (BFS từ node gốc = node nhỏ nhất của mỗi đồ thị)
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    first_node = pd.Series(np.arange(len(node_graph))).groupby(node_graph).min()
    depth = []
    for g in range(n_graphs):
        root = first_node[g]
        seen, q, d = {root: 0}, deque([root]), 0
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in seen:
                    seen[v] = seen[u] + 1
                    d = max(d, seen[v])
                    q.append(v)
        depth.append(d)
    df["depth"] = depth
    df.to_csv(os.path.join(ROOT, name, "graph_meta.csv"), index=False)
    return df


def report(name, df, sizes):
    L = [f"## {name}", ""]
    L += [f"- {len(df)} đồ thị, {int(df['n_nodes'].sum())} node, {int(df['n_edges'].sum())} cạnh.",
          f"- Kích thước đồ thị: median {df['n_nodes'].median():.0f} node, max {df['n_nodes'].max()}; "
          f"độ sâu median {df['depth'].median():.0f}, max {df['depth'].max()}.", ""]
    L += ["| Split | Số đồ thị | Fake (1) | Real (0) | Node median | Retweet đầu tiên (min → max) |",
          "|---|---|---|---|---|---|"]
    for s in ["train", "val", "test"]:
        d = df[df["split"] == s]
        L.append(f"| {s} | {len(d)} | {(d['label'] == 1).sum()} | {(d['label'] == 0).sum()} | "
                 f"{d['n_nodes'].median():.0f} | {d['first_retweet_date'].min()} → {d['first_retweet_date'].max()} |")
    L += ["", "**Nhãn theo năm (retweet đầu tiên)** — kiểm tra thời gian có phải confounder không:", "",
          "| Năm | Fake | Real | % Fake |", "|---|---|---|---|"]
    yr = pd.to_datetime(df["first_retweet"], unit="s").dt.year
    ct = pd.crosstab(yr, df["label"])
    for y, r in ct.iterrows():
        f, rl = r.get(1, 0), r.get(0, 0)
        L.append(f"| {int(y)} | {f} | {rl} | {f / (f + rl) * 100:.0f}% |")
    L += ["", "**Kích thước đồ thị theo nhãn** (lối tắt cấu trúc tiềm năng):", "",
          "| Nhãn | Node median | Độ sâu median |", "|---|---|---|"]
    for lab, nm in [(1, "Fake"), (0, "Real")]:
        d = df[df["label"] == lab]
        L.append(f"| {nm} | {d['n_nodes'].median():.0f} | {d['depth'].median():.0f} |")
    L += ["", "Số chiều feature PyG: " + ", ".join(f"{f} {sizes[(f, 'train')][1]}" for f in FEATURES), ""]
    return L


def main():
    L = ["# UPFD — báo cáo dữ liệu", "",
         "Nhãn theo quy ước UPFD: 1 = Fake, 0 = Real. Chia train/val/test gốc của tác giả (20/10/70).", ""]
    for name in ["politifact", "gossipcop"]:
        if not unzip(name):
            continue
        sizes = load_pyg(name)
        df = graph_meta(name)
        L += report(name, df, sizes)
        print(f"{name}: {len(df)} graphs ok")
    with open(os.path.join(REPORT_DIR, "data_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
