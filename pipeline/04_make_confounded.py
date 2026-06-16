"""
04_make_confounded.py
==========================
Build a CONTROLLED confounding-shift benchmark (ColoredMNIST-style) to demonstrate
the causal model's robustness to a spurious confounder, per PRD section 1.1.

Idea:
  - Replace the real subreddit with a SYNTHETIC binary confounder `env` in {0,1}
    realised as two subreddits: spur_fakebias / spur_realbias.
  - TRAIN / VAL / SEEN-TEST:  env is strongly correlated with the label (rho=0.9).
        -> subreddit fake_ratio (computed on train) becomes a near-perfect shortcut.
  - OOD TEST (is_ood=True):    correlation is REVERSED (rho=0.1).
        -> a model that relies on the subreddit confounder will FLIP to wrong labels;
           a model that uses content (causal) stays robust.

Content (mpnet text + CLIP image + scalars + domain/user) is UNCHANGED — only the
subreddit assignment is manipulated. Reuses existing embeddings (no re-download).

Output: data/processed_confounded/  (drop-in for 03/04 via GNN_INPUT_DIR)
"""
import os, shutil
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

SRC = os.path.join("data", "processed")
DST = os.path.join("data", "processed_confounded")
RHO_TRAIN = 0.9   # P(env signals true label) in train/val/seen-test
RHO_OOD   = 0.1   # reversed correlation on OOD test
os.makedirs(DST, exist_ok=True)

# ---- 1. Load posts (order defines embedding alignment) ----
posts = pd.read_csv(os.path.join(SRC, "posts_enriched.csv"))
n = len(posts)
fake = (posts["label_2way"] == 0).values           # 0 = Fake, 1 = Real
split = posts["split"].values
is_ood_existing = posts["is_ood"].fillna(False).astype(bool).values if "is_ood" in posts else np.zeros(n, bool)

# ---- 2. Assign synthetic env (the confounder) ----
# env==1 -> "fake-biased" subreddit. In train, env tracks `fake` with prob RHO_TRAIN.
# Reversed-correlation OOD test = the existing OOD posts (held-out) + seen-test stays correlated.
reversed_mask = is_ood_existing                     # OOD posts get reversed correlation
rho = np.where(reversed_mask, RHO_OOD, RHO_TRAIN)
signal = rng.random(n) < rho                        # True -> env matches fake flag
env = np.where(signal, fake, ~fake).astype(int)     # 1 if behaves-fake-biased else 0

sub_names = np.where(env == 1, "spur_fakebias", "spur_realbias")
posts["subreddit"] = sub_names
# keep split; OOD posts remain test+is_ood (reversed), seen-test stays correlated
posts["is_ood"] = reversed_mask

# ---- 3. Recompute subreddit fake_ratio from TRAIN assignment only ----
train_df = posts[posts["split"] == "train"]
sub_rows = []
for s in ["spur_fakebias", "spur_realbias"]:
    tr = train_df[train_df["subreddit"] == s]
    fr = float((tr["label_2way"] == 0).mean()) if len(tr) else 0.5   # P(fake|env)
    sub_rows.append({"sub_id": f"sub_{s}", "name": s,
                     "post_count": int(len(tr)),
                     "fake_ratio_real": fr,
                     "avg_score": float(tr["score"].mean()) if len(tr) else 0.0})
subs = pd.DataFrame(sub_rows)
print("Synthetic subreddit fake_ratio (train):")
print(subs[["name", "post_count", "fake_ratio_real"]].to_string(index=False))

# Report realised correlation
def corr(mask):
    e = env[mask]; f = fake[mask].astype(int)
    return float(np.mean(e == f))
tr_m = (split == "train"); seen_m = (split == "test") & ~reversed_mask; ood_m = reversed_mask
print(f"\nP(env==fake): train={corr(tr_m):.2f} | seen-test={corr(seen_m):.2f} | OOD-test={corr(ood_m):.2f}")
print(f"Split sizes: train={tr_m.sum()} val={(split=='val').sum()} seen-test={seen_m.sum()} OOD-test={ood_m.sum()}")

# ---- 4. Write modified CSVs; reuse everything else ----
# posts_enriched.csv (with new subreddit col) + posts.csv (is_ood)
posts.to_csv(os.path.join(DST, "posts_enriched.csv"), index=False)
posts_base = pd.read_csv(os.path.join(SRC, "posts.csv"))
posts_base["is_ood"] = reversed_mask
posts_base.to_csv(os.path.join(DST, "posts.csv"), index=False)

# subreddits_enriched.csv (synthetic) — keep 3 feature cols the model expects
subs.to_csv(os.path.join(DST, "subreddits_enriched.csv"), index=False)

# posted_in.csv  (Post -> synthetic Subreddit)
posted_in = pd.DataFrame({"post_id": posts["post_id"].values,
                          "sub_id": [f"sub_{s}" for s in sub_names]})
posted_in.to_csv(os.path.join(DST, "posted_in.csv"), index=False)

# member_of.csv (User -> synthetic Subreddit): rebuild from posted_by mapping
posted_by = pd.read_csv(os.path.join(SRC, "posted_by.csv"))
pid2sub = dict(zip(posts["post_id"].astype(str), [f"sub_{s}" for s in sub_names]))
mo = posted_by.copy()
mo["sub_id"] = mo["post_id"].astype(str).map(pid2sub)
mo = mo[["user_id", "sub_id"]].dropna().drop_duplicates()
mo["activity_level"] = 1
mo.to_csv(os.path.join(DST, "member_of.csv"), index=False)

# Copy through unchanged artifacts (content + other node types + embeddings)
for f in ["users_enriched.csv", "domains_enriched.csv", "images_enriched.csv",
          "posted_by.csv", "links_to.csv", "has_image.csv",
          "post_embeddings.npy", "post_fastrp.npy", "image_embeddings.npy",
          "clip_cons.npy"]:
    src_f = os.path.join(SRC, f)
    if os.path.exists(src_f):
        shutil.copy(src_f, os.path.join(DST, f))
    else:
        print(f"  [skip] {f} chưa tồn tại (chạy 03_clip_consistency.py nếu cần)")

print(f"\nWrote confounding-shift dataset to {DST}/  (reused mpnet/CLIP embeddings)")
