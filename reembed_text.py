"""
reembed_text.py
===============
Re-encode post titles with a stronger sentence encoder (default: all-mpnet-base-v2, 768-d)
to improve content understanding for OOD misinformation detection.

Reads titles from data/processed/posts_enriched.csv in row order (aligned with how
build_heterodata() consumes post_embeddings.npy) and overwrites post_embeddings.npy.
The old MiniLM embedding is preserved separately as a backup by the caller.
"""

import os
import sys
import numpy as np
import pandas as pd

INPUT_DIR = os.path.join("data", "processed")
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "all-mpnet-base-v2"


def main():
    posts = pd.read_csv(os.path.join(INPUT_DIR, "posts_enriched.csv"))
    titles = posts["title"].fillna("").astype(str).tolist()
    print(f"Encoding {len(titles)} titles with '{MODEL_NAME}'...")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(titles, show_progress_bar=True, batch_size=64,
                       convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype(np.float32)
    print(f"Embedding shape: {emb.shape}")

    out_path = os.path.join(INPUT_DIR, "post_embeddings.npy")
    np.save(out_path, emb)
    print(f"Saved {out_path} (dim={emb.shape[1]})")


if __name__ == "__main__":
    main()
