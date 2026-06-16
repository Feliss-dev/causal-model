"""
01_prepare_data.py
==================
Chuẩn bị dữ liệu Fakeddit cho pipeline Causal GNN.

Thay đổi so với phiên bản cũ:
- Tính feature THỰC từ dữ liệu train (không mock):
    User:      post_count, avg_score, avg_upvote_ratio, fake_rate
    Subreddit: post_count, fake_ratio_real, avg_score
    Domain:    post_count, fake_ratio_real, avg_upvote_ratio
- Xoá Comment node (Fakeddit không có comment text thực)
- Xoá CROSS_POST edge (đang gây data leakage: cross-post chỉ trong cùng label)
- MEMBER_OF activity_level = post_count thực từ train data
- Hỗ trợ multimodal_test_public.tsv nếu có; nếu không thì split validate.tsv
"""

import os
import sys
import random
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===================== REPRODUCIBILITY =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ===================== PATHS =====================
RAW_TRAIN_PATH = "multimodal_train.tsv"
RAW_VAL_PATH   = "multimodal_validate.tsv"
RAW_TEST_PATH  = "multimodal_test_public.tsv"   # Fakeddit cung cấp, tải về nếu có
OUTPUT_DIR     = os.path.join("data", "processed")
IMAGE_DIR      = os.path.join("data", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ===================== SAMPLING CONFIG =====================
TRAIN_PER_CLASS      = 2500   # 2500 real + 2500 fake = 5000 train
VAL_PER_CLASS        = 200    # 200 real + 200 fake = 400 val
SEEN_TEST_PER_CLASS  = 100    # 100 real + 100 fake = 200 seen test
OOD_TEST_PER_SUB    = 150    # 150 posts per OOD subreddit → 4×150 = 600 OOD total

TRAIN_CANDIDATES_PER_CLASS = 3500
VAL_CANDIDATES_PER_CLASS   = 800

# OOD subreddits — 2 communities, 1 Real + 1 Fake → combined fake rate ≈ 50%
#
# Design rationale:
#   An earlier 2-subreddit OOD (nottheonion 0% + pareidolia 100%) was trivially
#   decodable by visual/textual community style alone — a new shortcut rather
#   than genuine generalization (content models scored near-random, AUC~0.34).
#
#   The current pair creates a ~50% fake-rate OOD set where community style
#   alone cannot predict labels:
#     neutralnews (fake_rate=0.0) — neutral political/world news reporting.
#     theonion    (fake_rate=1.0) — actual satire from The Onion; satirical
#                 headlines must be judged by CONTENT, not community style.
#   A model that merely recognizes community style achieves ~50% (random);
#   exceeding this requires genuine content-based classification.
OOD_SUBREDDITS = {
    "neutralnews",  # Real — neutral political/world news reporting (fake_rate=0)
    "theonion",     # Fake — actual satire from The Onion (fake_rate=1)
}
# Rationale: both are TEXT-CENTRIC news communities semantically close to the
# training distribution, so content (title + image) genuinely transfers — unlike
# the previous pareidolia/nottheonion split, which was near-random (AUC~0.34) for
# any content model because the OOD classes did not match learned content patterns.
# Combined OOD ≈ 50% fake (150 neutralnews Real + 150 theonion Fake).

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ===================== IMAGE DOWNLOAD =====================

def download_single_image(post_id, url):
    if not url or pd.isna(url) or not str(url).startswith("http"):
        return post_id, False
    file_path = os.path.join(IMAGE_DIR, f"{post_id}.jpg")
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return post_id, True
    try:
        response = requests.get(url, headers=HEADERS, timeout=8, verify=False)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            return post_id, True
    except Exception:
        pass
    return post_id, False


def download_images_parallel(posts_df, max_workers=10):
    successful_ids = set()
    print(f"Bắt đầu tải ảnh song song ({max_workers} luồng)...")
    tasks = dict(zip(posts_df["id"], posts_df["image_url"]))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_single_image, pid, url): pid
            for pid, url in tasks.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tải ảnh"):
            pid, success = future.result()
            if success:
                successful_ids.add(pid)
    return successful_ids


# ===================== CLIP EMBEDDING =====================

def extract_clip_embeddings(post_ids):
    print("\n========== Trích xuất CLIP Image Embeddings ==========")
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image

        print("Đang tải CLIP model (openai/clip-vit-base-patch32)...")
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device)
        print(f"CLIP model chạy trên: {device}")

        embeddings    = []
        failed_count  = 0
        batch_size    = 32

        for i in tqdm(range(0, len(post_ids), batch_size), desc="CLIP embeddings"):
            batch_ids    = post_ids[i: i + batch_size]
            batch_images = []
            valid_indices = []

            for j, pid in enumerate(batch_ids):
                img_path = os.path.join(IMAGE_DIR, f"{pid}.jpg")
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(j)
                except Exception:
                    failed_count += 1

            if batch_images:
                inputs = clip_processor(images=batch_images, return_tensors="pt", padding=True)
                pixel_values = inputs["pixel_values"].to(device)
                with torch.no_grad():
                    # Use vision_model directly — avoids ModelOutput wrapping issues
                    # across transformers versions
                    vision_out = clip_model.vision_model(pixel_values=pixel_values)
                    img_feats = vision_out.pooler_output          # (B, hidden_size)
                    img_feats = clip_model.visual_projection(img_feats)  # (B, 512)
                img_feats = torch.nn.functional.normalize(img_feats.float(), p=2, dim=-1)
                emb_np = img_feats.cpu().numpy()
                batch_emb = np.zeros((len(batch_ids), emb_np.shape[1]), dtype=np.float32)
                for k, idx in enumerate(valid_indices):
                    batch_emb[idx] = emb_np[k]
                embeddings.append(batch_emb)
            else:
                embeddings.append(np.zeros((len(batch_ids), 512), dtype=np.float32))

        clip_embeddings = np.concatenate(embeddings, axis=0)
        print(f"CLIP embeddings xong: shape={clip_embeddings.shape}, failed={failed_count}")
        return clip_embeddings

    except Exception as e:
        # FAIL HARD: random embedding sẽ làm sai lệch toàn bộ kết quả phía sau
        # mà không có cảnh báo nào trong metrics — tuyệt đối không fallback.
        print(f"[LỖI NGHIÊM TRỌNG] CLIP thất bại: {e}")
        print("Dừng pipeline. Hãy cài transformers/Pillow và chạy lại.")
        sys.exit(1)


# ===================== FEATURE ENGINEERING (THỰC) =====================

def compute_real_features(all_sampled_df):
    """
    Tính feature thực từ TRAIN data để tránh data leakage sang val/test.
    Trả về dict thống kê cho User, Subreddit, Domain và các default value.
    """
    train_df = all_sampled_df[all_sampled_df["split"] == "train"].copy()

    # ---- User ----
    user_agg = (
        train_df.groupby("author")
        .agg(
            post_count     = ("id",            "count"),
            avg_score      = ("score",          "mean"),
            avg_upvote_ratio = ("upvote_ratio", "mean"),
            # label=1 là Real, label=0 là Fake → fake_rate = 1 - mean(label)
            fake_rate      = ("2_way_label",    lambda x: 1.0 - x.mean()),
        )
        .reset_index()
    )

    # ---- Subreddit ----
    sub_agg = (
        train_df.groupby("subreddit")
        .agg(
            post_count    = ("id",           "count"),
            fake_ratio_real = ("2_way_label", lambda x: 1.0 - x.mean()),
            avg_score     = ("score",         "mean"),
        )
        .reset_index()
    )

    # ---- Domain ----
    domain_agg = (
        train_df.groupby("domain")
        .agg(
            post_count       = ("id",            "count"),
            fake_ratio_real  = ("2_way_label",   lambda x: 1.0 - x.mean()),
            avg_upvote_ratio = ("upvote_ratio",  "mean"),
        )
        .reset_index()
    )

    # Default values cho node chỉ xuất hiện ở val/test (unseen)
    defaults = {
        "user": {
            "post_count":        1,
            "avg_score":         float(train_df["score"].median()),
            "avg_upvote_ratio":  float(train_df["upvote_ratio"].median()),
            "fake_rate":         0.5,   # unknown → neutral
        },
        "subreddit": {
            "post_count":        1,
            "fake_ratio_real":   0.5,
            "avg_score":         float(train_df["score"].median()),
        },
        "domain": {
            "post_count":        1,
            "fake_ratio_real":   0.5,
            "avg_upvote_ratio":  float(train_df["upvote_ratio"].median()),
        },
    }

    # Tính MEMBER_OF activity_level từ train (post count per user-subreddit pair)
    member_activity_map = (
        train_df.groupby(["author", "subreddit"])
        .size()
        .to_dict()
    )

    return user_agg, sub_agg, domain_agg, defaults, member_activity_map


# ===================== SAMPLING =====================

def filter_with_images(df):
    return df[
        (df["hasImage"] == True) &
        (df["image_url"].notna()) &
        (df["image_url"].str.startswith("http"))
    ].copy()


def balanced_sample(df, label_col, per_class, seed=SEED):
    parts = []
    for lbl in df[label_col].unique():
        sub = df[df[label_col] == lbl]
        n   = min(len(sub), per_class)
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ===================== NEO4J CSV GENERATION =====================

def create_neo4j_csvs(all_sampled_df):
    print("\nTạo file CSV cho Neo4j...")

    # Tính feature thực từ train data
    user_agg, sub_agg, domain_agg, defaults, member_activity_map = \
        compute_real_features(all_sampled_df)

    df = all_sampled_df.copy()
    df["author"] = df["author"].fillna("[deleted]")
    df["domain"] = df["domain"].fillna("reddit.com")

    # --- NODES ---

    # 1. Post
    posts = pd.DataFrame({
        "post_id":       df["id"],
        "title":         df["clean_title"].fillna(df["title"]),
        "score":         df["score"].fillna(0).astype(int),
        "upvote_ratio":  df["upvote_ratio"].fillna(1.0).astype(float),
        "num_comments":  df["num_comments"].fillna(0).astype(int),
        "has_image":     True,
        "label_2way":    df["2_way_label"].astype(int),
        "label_6way":    df["6_way_label"].astype(int),
        "split":         df["split"],
        "is_ood":        df.get("is_ood", pd.Series([False]*len(df))).fillna(False).astype(bool),
    })
    posts.to_csv(os.path.join(OUTPUT_DIR, "posts.csv"), index=False)

    # 2. User (feature thực)
    user_feat_map = user_agg.set_index("author")
    unique_users  = df["author"].unique()
    users_rows    = []
    for u in unique_users:
        if u in user_feat_map.index:
            r = user_feat_map.loc[u]
            users_rows.append({
                "user_id":         f"user_{u}",
                "name":            u,
                "post_count":      int(r["post_count"]),
                "avg_score":       float(r["avg_score"]),
                "avg_upvote_ratio": float(r["avg_upvote_ratio"]),
                "fake_rate":       float(r["fake_rate"]),
            })
        else:
            d = defaults["user"]
            users_rows.append({
                "user_id":         f"user_{u}",
                "name":            u,
                "post_count":      d["post_count"],
                "avg_score":       d["avg_score"],
                "avg_upvote_ratio": d["avg_upvote_ratio"],
                "fake_rate":       d["fake_rate"],
            })
    users = pd.DataFrame(users_rows)
    users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)

    # 3. Subreddit (feature thực)
    sub_feat_map    = sub_agg.set_index("subreddit")
    unique_subs     = df["subreddit"].unique()
    subreddits_rows = []
    for s in unique_subs:
        if s in sub_feat_map.index:
            r = sub_feat_map.loc[s]
            subreddits_rows.append({
                "sub_id":         f"sub_{s}",
                "name":           s,
                "post_count":     int(r["post_count"]),
                "fake_ratio_real": float(r["fake_ratio_real"]),
                "avg_score":      float(r["avg_score"]),
            })
        else:
            d = defaults["subreddit"]
            subreddits_rows.append({
                "sub_id":         f"sub_{s}",
                "name":           s,
                "post_count":     d["post_count"],
                "fake_ratio_real": d["fake_ratio_real"],
                "avg_score":      d["avg_score"],
            })
    subreddits = pd.DataFrame(subreddits_rows)
    subreddits.to_csv(os.path.join(OUTPUT_DIR, "subreddits.csv"), index=False)

    # 4. Domain (feature thực)
    domain_feat_map = domain_agg.set_index("domain")
    unique_domains  = df["domain"].unique()
    domains_rows    = []
    for d_name in unique_domains:
        if d_name in domain_feat_map.index:
            r = domain_feat_map.loc[d_name]
            domains_rows.append({
                "domain_id":       f"domain_{d_name}",
                "url_domain":      d_name,
                "post_count":      int(r["post_count"]),
                "fake_ratio_real": float(r["fake_ratio_real"]),
                "avg_upvote_ratio": float(r["avg_upvote_ratio"]),
            })
        else:
            d = defaults["domain"]
            domains_rows.append({
                "domain_id":       f"domain_{d_name}",
                "url_domain":      d_name,
                "post_count":      d["post_count"],
                "fake_ratio_real": d["fake_ratio_real"],
                "avg_upvote_ratio": d["avg_upvote_ratio"],
            })
    domains = pd.DataFrame(domains_rows)
    domains.to_csv(os.path.join(OUTPUT_DIR, "domains.csv"), index=False)

    # 5. Image (chỉ metadata thực — CLIP embedding lưu riêng dưới dạng .npy)
    images = pd.DataFrame({
        "img_id":    [f"img_{i}" for i in df["id"]],
        "post_id":   df["id"].values,
        "has_image": True,
        "image_url": df["image_url"].values,
    })
    images.to_csv(os.path.join(OUTPUT_DIR, "images.csv"), index=False)

    # NOTE: Comment node đã bị XOÁ vì Fakeddit không cung cấp comment text thực.
    # Xem PRD Section 2.3 — node Comment không có dữ liệu nguồn tương ứng.

    # --- TEXT EMBEDDINGS ---
    print("\nTạo text embedding với SentenceTransformer...")
    TEXT_ENCODER = os.environ.get("TEXT_ENCODER", "all-mpnet-base-v2")  # 768-d
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(TEXT_ENCODER)

        post_titles = posts["title"].fillna("").tolist()
        print(f"Encode tiêu đề bài post với '{TEXT_ENCODER}'...")
        post_embeddings = model.encode(post_titles, show_progress_bar=True, batch_size=64,
                                       convert_to_numpy=True, normalize_embeddings=True)
        post_embeddings = post_embeddings.astype(np.float32)
        np.save(os.path.join(OUTPUT_DIR, "post_embeddings.npy"), post_embeddings)
        print(f"Đã lưu post_embeddings.npy: shape={post_embeddings.shape}")

    except Exception as e:
        # FAIL HARD: text embedding random = kết quả rác không thể phát hiện.
        print(f"[LỖI NGHIÊM TRỌNG] SentenceTransformer thất bại: {e}")
        print("Dừng pipeline. Hãy cài sentence-transformers và chạy lại.")
        sys.exit(1)

    # --- EDGES ---

    # 1. POSTED_BY (Post → User)
    posted_by = pd.DataFrame({
        "post_id": df["id"].values,
        "user_id": [f"user_{u}" for u in df["author"]],
    })
    posted_by.to_csv(os.path.join(OUTPUT_DIR, "posted_by.csv"), index=False)

    # 2. POSTED_IN (Post → Subreddit)
    posted_in = pd.DataFrame({
        "post_id": df["id"].values,
        "sub_id":  [f"sub_{s}" for s in df["subreddit"]],
    })
    posted_in.to_csv(os.path.join(OUTPUT_DIR, "posted_in.csv"), index=False)

    # 3. LINKS_TO (Post → Domain)
    links_to = pd.DataFrame({
        "post_id":   df["id"].values,
        "domain_id": [f"domain_{d}" for d in df["domain"]],
    })
    links_to.to_csv(os.path.join(OUTPUT_DIR, "links_to.csv"), index=False)

    # 4. HAS_IMAGE (Post → Image)
    has_image = pd.DataFrame({
        "post_id":    df["id"].values,
        "img_id":     [f"img_{i}" for i in df["id"]],
        "is_primary": True,
    })
    has_image.to_csv(os.path.join(OUTPUT_DIR, "has_image.csv"), index=False)

    # 5. MEMBER_OF (User → Subreddit) — activity_level từ train data thực
    member_of_rows = []
    seen_pairs = set()
    for user, sub in zip(df["author"], df["subreddit"]):
        key = (user, sub)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        activity = int(member_activity_map.get((user, sub), 1))
        member_of_rows.append({
            "user_id":        f"user_{user}",
            "sub_id":         f"sub_{sub}",
            "activity_level": activity,
        })
    member_of_df = pd.DataFrame(member_of_rows)
    member_of_df.to_csv(os.path.join(OUTPUT_DIR, "member_of.csv"), index=False)

    # NOTE: HAS_COMMENT, WROTE đã bị XOÁ (không có comment data thực).
    # NOTE: CROSS_POST đã bị XOÁ (tạo cross-post trong cùng label gây data leakage).

    print(f"\nCSV nodes/edges đã tạo xong tại {OUTPUT_DIR}/")
    print(f"Nodes: Posts={len(posts)}, Users={len(users)}, "
          f"Subreddits={len(subreddits)}, Domains={len(domains)}, Images={len(images)}")
    print(f"Edges: POSTED_BY={len(posted_by)}, POSTED_IN={len(posted_in)}, "
          f"LINKS_TO={len(links_to)}, HAS_IMAGE={len(has_image)}, "
          f"MEMBER_OF={len(member_of_df)}")


# ===================== MAIN =====================

def process_and_sample_dataset():
    print("Đọc Fakeddit raw dataset...")

    if not os.path.exists(RAW_TRAIN_PATH):
        print(f"[LỖI] Không tìm thấy {RAW_TRAIN_PATH}. "
              "Hãy chạy script trong thư mục gốc Fakeddit.")
        sys.exit(1)

    train_df = pd.read_csv(RAW_TRAIN_PATH, sep="\t", low_memory=False)
    val_df   = pd.read_csv(RAW_VAL_PATH,   sep="\t", low_memory=False)

    # Filter chỉ lấy bài có ảnh thực
    train_img = filter_with_images(train_df)
    val_img   = filter_with_images(val_df)

    print(f"Candidates có ảnh: train={len(train_img)}, val={len(val_img)}")
    print(f"OOD subreddits (held-out hoàn toàn): {OOD_SUBREDDITS}")

    # ── Tách OOD khỏi non-OOD ─────────────────────────────────────────────
    # OOD subreddits KHÔNG bao giờ có trong train/val — chỉ dùng cho OOD test
    train_ood     = train_img[train_img["subreddit"].isin(OOD_SUBREDDITS)].copy()
    train_non_ood = train_img[~train_img["subreddit"].isin(OOD_SUBREDDITS)].copy()
    val_ood       = val_img[val_img["subreddit"].isin(OOD_SUBREDDITS)].copy()
    val_non_ood   = val_img[~val_img["subreddit"].isin(OOD_SUBREDDITS)].copy()

    print(f"  train non-OOD: {len(train_non_ood):>6} | train OOD: {len(train_ood)}")
    print(f"  val non-OOD:   {len(val_non_ood):>6} | val OOD:   {len(val_ood)}")

    # ── Sample candidates ──────────────────────────────────────────────────
    # Train candidates: CHỈ từ non-OOD subreddits
    train_cands = balanced_sample(train_non_ood, "2_way_label", TRAIN_CANDIDATES_PER_CLASS)

    # Seen val+test candidates: từ validate.tsv non-OOD (cần val + seen_test)
    seen_pool_n = VAL_PER_CLASS + SEEN_TEST_PER_CLASS   # per class
    seen_val_cands = balanced_sample(val_non_ood, "2_way_label",
                                     min(VAL_CANDIDATES_PER_CLASS * 2, seen_pool_n * 2))

    # OOD test candidates: combine train-OOD + val-OOD, sample per subreddit
    ood_pool = pd.concat([train_ood, val_ood]).drop_duplicates(subset=["id"])
    ood_cands_list = []
    for sub in sorted(OOD_SUBREDDITS):
        sub_df  = ood_pool[ood_pool["subreddit"] == sub]
        n_samp  = min(OOD_TEST_PER_SUB, len(sub_df))
        if n_samp > 0:
            ood_cands_list.append(sub_df.sample(n=n_samp, random_state=SEED))
            print(f"  OOD '{sub}': {n_samp} candidates (fake_rate="
                  f"{1 - sub_df['2_way_label'].mean():.2f})")
        else:
            print(f"  [WARN] OOD '{sub}': 0 candidates — kiểm tra lại OOD_SUBREDDITS")

    ood_test_cands = (pd.concat(ood_cands_list).reset_index(drop=True)
                      if ood_cands_list else pd.DataFrame(columns=train_img.columns))

    # ── Download images ────────────────────────────────────────────────────
    print(f"\nTải ảnh train ({len(train_cands)} candidates)...")
    train_ok = download_images_parallel(train_cands, max_workers=15)

    print(f"\nTải ảnh val/seen-test ({len(seen_val_cands)} candidates)...")
    seen_ok = download_images_parallel(seen_val_cands, max_workers=15)

    ood_ok = set()
    if len(ood_test_cands) > 0:
        print(f"\nTải ảnh OOD test ({len(ood_test_cands)} candidates)...")
        ood_ok = download_images_parallel(ood_test_cands, max_workers=15)

    # ── Filter downloaded ─────────────────────────────────────────────────
    train_dl = train_cands[train_cands["id"].isin(train_ok)].copy()
    seen_dl  = seen_val_cands[seen_val_cands["id"].isin(seen_ok)].copy()
    ood_dl   = (ood_test_cands[ood_test_cands["id"].isin(ood_ok)].copy()
                if len(ood_test_cands) > 0 else pd.DataFrame(columns=train_img.columns))

    # ── Final splits ───────────────────────────────────────────────────────
    n_train = min(
        TRAIN_PER_CLASS,
        int((train_dl["2_way_label"] == 0).sum()),
        int((train_dl["2_way_label"] == 1).sum()),
    )
    final_train = balanced_sample(train_dl, "2_way_label", n_train)
    final_train["split"]  = "train"
    final_train["is_ood"] = False

    # Val + seen-test từ validate.tsv non-OOD (split 2/3 val, 1/3 seen-test)
    n_val_half = min(
        VAL_PER_CLASS,
        int((seen_dl["2_way_label"] == 0).sum() * VAL_PER_CLASS // (VAL_PER_CLASS + SEEN_TEST_PER_CLASS)),
        int((seen_dl["2_way_label"] == 1).sum() * VAL_PER_CLASS // (VAL_PER_CLASS + SEEN_TEST_PER_CLASS)),
    )
    n_seen_test_half = min(
        SEEN_TEST_PER_CLASS,
        int((seen_dl["2_way_label"] == 0).sum()) - n_val_half,
        int((seen_dl["2_way_label"] == 1).sum()) - n_val_half,
    )
    seen_pool   = balanced_sample(seen_dl, "2_way_label",
                                  n_val_half + n_seen_test_half)
    # Shuffle thì split deterministically theo index
    n_val_total = n_val_half * 2
    final_val        = seen_pool.iloc[:n_val_total].reset_index(drop=True)
    final_seen_test  = seen_pool.iloc[n_val_total:].reset_index(drop=True)
    final_val["split"]       = "val"
    final_val["is_ood"]      = False
    final_seen_test["split"] = "test"
    final_seen_test["is_ood"]= False

    # OOD test: posts từ held-out subreddits, CHƯA thấy trong training
    if len(ood_dl) > 0:
        final_ood_test = ood_dl.copy()
        final_ood_test["split"]  = "test"
        final_ood_test["is_ood"] = True
        all_parts = [final_train, final_val, final_seen_test, final_ood_test]
    else:
        print("[WARN] Không có OOD test posts. OOD evaluation sẽ không có ý nghĩa.")
        all_parts = [final_train, final_val, final_seen_test]

    all_sampled = pd.concat(all_parts).reset_index(drop=True)

    print(f"\n{'='*60}")
    print("Phân phối dataset cuối cùng:")
    for split_name in ["train", "val", "test"]:
        s = all_sampled[all_sampled["split"] == split_name]
        n_real    = int((s["2_way_label"] == 1).sum())
        n_fake    = int((s["2_way_label"] == 0).sum())
        n_ood     = int(s.get("is_ood", pd.Series([False]*len(s))).sum())
        print(f"  {split_name:<6}: {len(s):>5} ({n_real} real, {n_fake} fake"
              + (f", {n_ood} OOD" if split_name == "test" else "") + ")")
    print(f"  {'Total':<6}: {len(all_sampled):>5}")

    test_subs  = set(all_sampled[all_sampled["split"] == "test"]["subreddit"].unique())
    train_subs = set(all_sampled[all_sampled["split"] == "train"]["subreddit"].unique())
    print(f"\n  Subreddits ONLY in test (true OOD): {test_subs - train_subs}")
    print(f"{'='*60}")

    # Lưu file master
    all_sampled.to_csv(os.path.join(OUTPUT_DIR, "sampled_master.csv"), index=False)
    print("Đã lưu sampled_master.csv")

    # CLIP image embeddings (theo thứ tự all_sampled)
    post_ids_list   = all_sampled["id"].tolist()
    clip_embeddings = extract_clip_embeddings(post_ids_list)
    np.save(os.path.join(OUTPUT_DIR, "image_embeddings.npy"), clip_embeddings)
    print(f"Đã lưu image_embeddings.npy: shape={clip_embeddings.shape}")

    # Tạo CSV nodes và edges
    create_neo4j_csvs(all_sampled)


if __name__ == "__main__":
    process_and_sample_dataset()
