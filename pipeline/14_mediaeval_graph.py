"""
14_mediaeval_graph.py
=====================
Dựng đồ thị dị thể MediaEval 2015 (Twitter) để HUẤN LUYỆN CausalHeteroGNN trên
chính MediaEval, chia theo sự kiện, và in báo cáo kiểm tra dữ liệu (Bước 0).

Đồ thị (xem results/mediaeval_diag/step0_report.md):
  Node:  Tweet  = mpnet(clean text) 768 + tweet_features 18 + CLIPcons 1
         User   = user_features 6 (friends, followers, ratio, listed, verified, num_tweets)
         Image  = CLIP ViT-B/32 512 (một node cho mỗi ảnh, dùng chung giữa các tweet)
         Event  = log(số tweet) 1  — confounder; KHÔNG dùng thống kê nhãn
  Cạnh:  Tweet-posted_by->User, Tweet-has_image->Image, Tweet-in_event->Event
         (+ rev_* do ToUndirected ở bước train). G_causal bỏ in_event/rev_in_event.
Nhãn:   humor gộp vào Fake (quy ước MediaEval 2015). 0 = Fake, 1 = Real.
Chia:   train = devset trừ Boston | val = Boston | test = testset.
        Tên thư mục ảnh dev (fakes/ reals/) chứa nhãn → chỉ map ảnh theo image_id.

Cách chạy (từ project root):
  $env:PYTHONUTF8 = "1"
  uv run python pipeline/14_mediaeval_graph.py
Output:
  data/mediaeval2015/graph/{graph.pt, tweets_meta.csv, *.npy cache}
  results/mediaeval_diag/step0_report.{md,json}
"""

import os
import re
import json
import numpy as np
import pandas as pd
import torch

ROOT      = os.environ.get("MEDIAEVAL_ROOT", "D:/MediaTwitter/image-verification-corpus/mediaeval2015")
DEV_IMG   = os.path.join("data", "mediaeval2015", "dev_images")   # giải nén từ MediaEval2015_DevSet_Images.rar
TEST_IMG  = os.path.join(ROOT, "testset", "MediaEval2015_TestSetImages", "TestSetImages")
GRAPH_DIR = os.path.join("data", "mediaeval2015", "graph")
DIAG_DIR  = os.path.join("results", "mediaeval_diag")
VAL_EVENTS = {"boston"}
VIDEO_EVENTS = {"syrianboy", "varoufakis"}          # testset: chỉ có file *_video.txt
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(DIAG_DIR, exist_ok=True)

TWEET_FEATS = ["num_words", "text_length", "contains_questmark", "num_questmark",
               "contains_exclammark", "num_exclammark", "contains_happyemo", "contains_sademo",
               "contains_firstorderpron", "contains_secondorderpron", "contains_thirdorderpron",
               "num_uppercasechars", "num_possentiwords", "num_negsentiwords", "num_mentions",
               "num_hashtags", "num_URLs", "num_retweets"]
USER_FEATS = ["num_friends", "num_followers", "folfriend_ratio", "times_listed",
              "is_verified", "num_tweets"]


# ===================== LOAD =====================

def load_tweets(part):
    df = pd.read_csv(os.path.join(ROOT, part, "tweets.txt"), sep="\t", dtype=str,
                     quoting=3, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["label_raw"] = df["label"].str.strip().str.lower()
    df = df[df["label_raw"].isin(["fake", "real", "humor"])].copy()
    df["label_2way"] = (df["label_raw"] == "real").astype(int)        # humor -> Fake
    df["image_list"] = df["imageId(s)"].apply(lambda s: [i.strip() for i in s.split(",") if i.strip()])
    df["event"] = df["image_list"].apply(lambda l: event_of(l[0]) if l else "unknown")
    df["part"] = part
    return df


def event_of(image_id):
    e = image_id.split("_")[0]
    return "sandy" if e.startswith("sandy") else e


def load_feature_table(part, name, cols):
    ext = "csv" if part == "devset" else "txt"
    df = pd.read_csv(os.path.join(ROOT, part, "DatasetFeatures", f"{name}.{ext}"),
                     dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    out = df[["tweet_id"]].copy()
    for c in cols:
        v = df[c].str.strip().str.lower().replace({"true": "1", "false": "0"})
        out[c] = pd.to_numeric(v, errors="coerce")
    return out.drop_duplicates("tweet_id")


def image_files():
    files = {}
    for root in (DEV_IMG, TEST_IMG):
        for dirpath, _, names in os.walk(root):
            for n in names:
                stem, ext = os.path.splitext(n)
                if ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
                    files[stem] = os.path.join(dirpath, n)
    return files


def dev_video_ids():
    """Ảnh dev có URL là video/GIF (theo devset/images.txt)."""
    # Một số dòng có tab thừa ở cuối -> chỉ đọc 2 cột đầu (image_id, image_url)
    img = pd.read_csv(os.path.join(ROOT, "devset", "images.txt"), sep="\t", dtype=str,
                      keep_default_na=False, usecols=[0, 1], quoting=3)
    pat = re.compile(r"youtube|youtu\.be|vine\.co|vimeo|\.gif\b", re.I)
    return {r["image_id"].strip() for _, r in img.iterrows() if pat.search(r["image_url"])}


def clean_tweet(t):
    """Giống 13_eval_mediaeval.clean_tweet (lowercase, bỏ URL/mention/ký tự)."""
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(r"@\w+:?", " ", t)
    t = t.replace("#", " ").replace("\\n", " ")
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ===================== EMBEDDINGS (cached) =====================

def cached(name, n, fn):
    path = os.path.join(GRAPH_DIR, name)
    if os.path.exists(path):
        arr = np.load(path)
        if len(arr) == n:
            return arr
    arr = fn()
    np.save(path, arr)
    return arr


def mpnet(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-mpnet-base-v2")
    return m.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
                    normalize_embeddings=True).astype(np.float32)


def clip_images(paths):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            v = model.vision_model(pixel_values=proc(images=[img], return_tensors="pt")["pixel_values"]).pooler_output
            out.append(model.visual_projection(v).numpy())
    return np.concatenate(out).astype(np.float32)


def clip_texts(texts):
    from transformers import CLIPModel, CLIPTokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    out = []
    for i in range(0, len(texts), 128):
        inp = tok(texts[i:i + 128], padding=True, truncation=True, max_length=77, return_tensors="pt")
        with torch.no_grad():
            pooled = model.text_model(**inp).pooler_output
            out.append(model.text_projection(pooled).numpy())
    return np.concatenate(out).astype(np.float32)


def detect_lang(texts):
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    langs = []
    for t in texts:
        t = re.sub(r"https?://\S+|@\w+|#|\bRT\b", " ", t).strip()
        try:
            langs.append(detect(t) if len(t) >= 3 else "und")
        except Exception:
            langs.append("und")
    return langs


def minmax(a, ref_mask=None):
    ref = a if ref_mask is None else a[ref_mask]
    mn, mx = ref.min(axis=0), ref.max(axis=0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    return ((a - mn) / rng).astype(np.float32)


# ===================== BUILD =====================

def main():
    tw = pd.concat([load_tweets("devset"), load_tweets("testset")], ignore_index=True)
    tw = tw.drop_duplicates("tweetId").reset_index(drop=True)
    tw["split"] = np.where(tw["part"] == "testset", "test",
                           np.where(tw["event"].isin(VAL_EVENTS), "val", "train"))
    n = len(tw)
    print(f"Tweets: {n} | split: {tw['split'].value_counts().to_dict()}")

    # ---- image status per tweet ----
    files = image_files()
    vids = dev_video_ids()
    def status(row):
        ids = row["image_list"]
        if any(i in files for i in ids):
            return "image"
        if any(("video" in i) or (i in vids) for i in ids) or row["event"] in VIDEO_EVENTS:
            return "video_gif"
        return "missing"
    tw["image_status"] = tw.apply(status, axis=1)
    tw["image_files"] = tw["image_list"].apply(lambda l: [i for i in l if i in files])
    tw["has_gif_file"] = tw["image_files"].apply(lambda l: any(files[i].lower().endswith(".gif") for i in l))

    # ---- features ----
    tw["clean_text"] = tw["tweetText"].apply(clean_tweet)
    text_emb = cached("text_mpnet.npy", n, lambda: mpnet(tw["clean_text"].tolist()))

    img_ids = sorted({i for l in tw["image_files"] for i in l})
    img_emb = cached("image_clip.npy", len(img_ids), lambda: clip_images([files[i] for i in img_ids]))
    img_idx = {i: k for k, i in enumerate(img_ids)}

    has_img = tw["image_status"].eq("image").values
    ctext = cached("text_clip.npy", n, lambda: clip_texts(tw["tweetText"].str.slice(0, 300).tolist()))
    ctext_n = ctext / np.linalg.norm(ctext, axis=1, keepdims=True)
    img_n = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    clip_mean = np.zeros((n, img_emb.shape[1]), np.float32)
    clipcons = np.zeros((n, 1), np.float32)
    for r, l in enumerate(tw["image_files"]):
        if l:
            ks = [img_idx[i] for i in l]
            clip_mean[r] = img_emb[ks].mean(0)
            clipcons[r, 0] = float((img_n[ks] @ ctext_n[r]).mean())

    tf = pd.concat([load_feature_table(p, "tweet_features", TWEET_FEATS) for p in ("devset", "testset")])
    tf = tw[["tweetId"]].merge(tf, left_on="tweetId", right_on="tweet_id", how="left")[TWEET_FEATS]
    t_missing = int(tf.isna().all(axis=1).sum())
    tfeat = minmax(np.log1p(tf.fillna(0).clip(lower=0).values.astype(np.float64)))

    uf = pd.concat([load_feature_table(p, "user_features", USER_FEATS) for p in ("devset", "testset")])
    uf = tw[["tweetId", "userId"]].merge(uf, left_on="tweetId", right_on="tweet_id", how="left")
    users = uf.groupby("userId")[USER_FEATS].mean().fillna(0)
    ufeat = minmax(np.log1p(users.clip(lower=0).values.astype(np.float64)))
    user_idx = {u: k for k, u in enumerate(users.index)}

    events = sorted(tw["event"].unique())
    ev_idx = {e: k for k, e in enumerate(events)}
    ev_counts = tw["event"].value_counts()
    efeat = minmax(np.log1p(np.array([[ev_counts[e]] for e in events], np.float64)))

    tweet_x = np.concatenate([text_emb, tfeat, clipcons], axis=1).astype(np.float32)

    # ---- edges ----
    e_user = np.array([[r, user_idx[u]] for r, u in enumerate(tw["userId"])]).T
    e_img = np.array([[r, img_idx[i]] for r, l in enumerate(tw["image_files"]) for i in l]).T
    e_ev = np.array([[r, ev_idx[e]] for r, e in enumerate(tw["event"])]).T

    train_events = sorted(tw.loc[tw["split"] == "train", "event"].unique())
    tr_ev_idx = {e: k for k, e in enumerate(train_events)}
    conf_label = np.array([tr_ev_idx.get(e, -1) if s == "train" else -1
                           for e, s in zip(tw["event"], tw["split"])])

    graph = {
        "x": {"Tweet": torch.tensor(tweet_x), "User": torch.tensor(ufeat),
              "Image": torch.tensor(img_emb), "Event": torch.tensor(efeat)},
        "edges": {("Tweet", "posted_by", "User"): torch.tensor(e_user, dtype=torch.long),
                  ("Tweet", "has_image", "Image"): torch.tensor(e_img, dtype=torch.long),
                  ("Tweet", "in_event", "Event"): torch.tensor(e_ev, dtype=torch.long)},
        "tweet_clip_mean": torch.tensor(clip_mean),
        "y": torch.tensor(tw["label_2way"].values, dtype=torch.long),
        "conf_label": torch.tensor(conf_label, dtype=torch.long),
        "train_mask": torch.tensor((tw["split"] == "train").values),
        "val_mask": torch.tensor((tw["split"] == "val").values),
        "test_mask": torch.tensor((tw["split"] == "test").values),
        "train_events": train_events, "events": events,
    }
    torch.save(graph, os.path.join(GRAPH_DIR, "graph.pt"))

    tw["lang"] = detect_lang(tw["tweetText"].tolist())
    meta_cols = ["tweetId", "userId", "event", "split", "label_raw", "label_2way",
                 "image_status", "lang"]
    tw[meta_cols].assign(image_ids=tw["imageId(s)"].str.strip()).to_csv(
        os.path.join(GRAPH_DIR, "tweets_meta.csv"), index=False)

    step0_report(tw, img_ids, e_user, e_img, e_ev, users, t_missing, tweet_x.shape[1])


# ===================== STEP 0 REPORT =====================

def step0_report(tw, img_ids, e_user, e_img, e_ev, users, t_missing, tweet_dim):
    R, L = {}, []
    lab = tw["label_2way"].map({0: "fake", 1: "real"})

    # 1. Event x nhãn
    L += ["# Bước 0 — Kiểm tra dữ liệu MediaEval 2015", "",
          "Nhãn 2 lớp: humor gộp vào Fake. Chia: train = devset trừ Boston, val = Boston, test = testset.", "",
          "## 1. Sự kiện × nhãn", ""]
    for name, parts in [("dev (train + val)", ["train", "val"]), ("test", ["test"])]:
        sub = tw[tw["split"].isin(parts)]
        ct = pd.crosstab(sub["event"], sub["label_raw"])
        for c in ["fake", "humor", "real"]:
            if c not in ct:
                ct[c] = 0
        ct = ct[["fake", "humor", "real"]]
        ct["n"] = ct.sum(axis=1)
        ct["pure_2way"] = ((ct["fake"] + ct["humor"] == 0) | (ct["real"] == 0))
        ct = ct.sort_values("n", ascending=False)
        n_pure = int(ct["pure_2way"].sum())
        pct_pure = float(ct.loc[ct["pure_2way"], "n"].sum() / ct["n"].sum())
        R[f"events_{name}"] = {"n_events": len(ct), "n_pure": n_pure, "pct_tweets_in_pure": pct_pure}
        L += [f"**{name}:** {len(ct)} sự kiện, {n_pure} sự kiện thuần một nhãn; "
              f"{pct_pure*100:.1f}% tweet nằm trong sự kiện thuần.", "",
              "| Sự kiện | fake | humor | real | tổng | thuần |", "|---|---|---|---|---|---|"]
        for e, r in ct.iterrows():
            L.append(f"| {e} | {r.fake} | {r.humor} | {r.real} | {r.n} | {'có' if r.pure_2way else 'không'} |")
        L.append("")

    # 2. Ảnh
    rows = [(i, s, y) for l, s, y in zip(tw["image_files"], tw["split"], tw["label_2way"]) for i in l]
    im = pd.DataFrame(rows, columns=["img", "split", "y"])
    per_img = im.groupby("img").agg(n=("y", "size"), pure=("y", lambda v: v.nunique() == 1))
    ids_ref = {i for l in tw["image_list"] for i in l}
    R["images"] = {"n_tweets": len(tw), "n_image_ids_referenced": len(ids_ref),
                   "n_image_files": len(img_ids),
                   "tweets_per_image_min": int(per_img["n"].min()),
                   "tweets_per_image_median": float(per_img["n"].median()),
                   "tweets_per_image_max": int(per_img["n"].max()),
                   "pct_images_label_pure": float(per_img["pure"].mean()),
                   "top5_images": per_img["n"].sort_values(ascending=False).head(5).to_dict()}
    i = R["images"]
    L += ["## 2. Ảnh", "",
          f"- {len(tw)} tweet, {i['n_image_ids_referenced']} image_id khác nhau được tham chiếu, "
          f"{i['n_image_files']} có file ảnh (thành node Image).",
          f"- Số tweet trên mỗi ảnh: min {i['tweets_per_image_min']}, median {i['tweets_per_image_median']:.0f}, "
          f"max {i['tweets_per_image_max']}.",
          f"- {i['pct_images_label_pure']*100:.1f}% ảnh có nhãn thuần (mọi tweet dùng ảnh đó cùng nhãn).",
          f"- 5 ảnh nhiều tweet nhất: " + ", ".join(f"{k} ({v})" for k, v in i["top5_images"].items()), ""]

    # 3. Ảnh thiếu / video
    st = pd.crosstab(tw["split"], tw["image_status"])
    R["image_status"] = st.to_dict()
    R["gif_tweets"] = int(tw["has_gif_file"].sum())
    L += ["## 3. Tweet thiếu ảnh / video / GIF", "",
          "| Tập | có ảnh | video/GIF (không có file) | thiếu file |", "|---|---|---|---|"]
    for s in ["train", "val", "test"]:
        r = st.loc[s] if s in st.index else {}
        L.append(f"| {s} | {r.get('image', 0)} | {r.get('video_gif', 0)} | {r.get('missing', 0)} |")
    L += ["", f"Tweet không có file ảnh nào sẽ có đặc trưng ảnh toàn 0 (CLIPcons = 0, không có cạnh has_image). "
          f"{R['gif_tweets']} tweet dùng file .gif (lấy khung đầu). {t_missing} tweet không có dòng tweet_features.", ""]

    # 4. Ngôn ngữ
    L += ["## 4. Ngôn ngữ (langdetect)", "", "| Tập | 5 ngôn ngữ nhiều nhất | % tiếng Anh |", "|---|---|---|"]
    R["lang"] = {}
    for s in ["train", "val", "test"]:
        vc = tw.loc[tw["split"] == s, "lang"].value_counts(normalize=True)
        R["lang"][s] = vc.head(10).round(4).to_dict()
        L.append(f"| {s} | " + ", ".join(f"{k} {v*100:.1f}%" for k, v in vc.head(5).items())
                 + f" | {vc.get('en', 0)*100:.1f}% |")
    L.append("")

    # 5. User
    deg = tw.groupby("userId").size()
    dev_users = set(tw.loc[tw["split"] != "test", "userId"])
    test_users = set(tw.loc[tw["split"] == "test", "userId"])
    R["users"] = {"n_users": int(len(deg)), "pct_one_tweet": float((deg == 1).mean()),
                  "max_tweets": int(deg.max()), "n_users_dev_and_test": len(dev_users & test_users)}
    u = R["users"]
    L += ["## 5. Bậc của node User", "",
          f"{u['n_users']} user; {u['pct_one_tweet']*100:.1f}% chỉ có 1 tweet; nhiều nhất {u['max_tweets']} tweet. "
          f"{u['n_users_dev_and_test']} user xuất hiện ở cả dev và test.", ""]

    # 6. Tỉ lệ lớp
    L += ["## 6. Tỉ lệ lớp", "", "| Tập | số tweet | Fake (gồm humor) | Real | % Fake |", "|---|---|---|---|---|"]
    R["class_ratio"] = {}
    for s in ["train", "val", "test"]:
        y = tw.loc[tw["split"] == s, "label_2way"]
        R["class_ratio"][s] = {"n": int(len(y)), "fake": int((y == 0).sum()), "real": int((y == 1).sum())}
        L.append(f"| {s} | {len(y)} | {(y == 0).sum()} | {(y == 1).sum()} | {(y == 0).mean()*100:.1f}% |")
    L.append("")

    # 7. Hàng xóm của Tweet
    n = len(tw)
    deg_user = np.bincount(e_user[1]); deg_img = np.bincount(e_img[1]); deg_ev = np.bincount(e_ev[1])
    one_hop_g = 1 + np.bincount(e_img[0], minlength=n) + 1
    one_hop_c = one_hop_g - 1
    two_hop = {"qua User": float(np.mean(deg_user[e_user[1]] - 1)),
               "qua Image": float(np.sum(deg_img[e_img[1]] - 1) / n),
               "qua Event": float(np.mean(deg_ev[e_ev[1]] - 1))}
    R["neighbors"] = {"one_hop_G": float(one_hop_g.mean()), "one_hop_G_causal": float(one_hop_c.mean()),
                      "two_hop_tweets": two_hop,
                      "two_hop_tweets_G": sum(two_hop.values()),
                      "two_hop_tweets_G_causal": two_hop["qua User"] + two_hop["qua Image"]}
    nb = R["neighbors"]
    L += ["## 7. Hàng xóm của node Tweet", "",
          "| | G | G_causal |", "|---|---|---|",
          f"| Hàng xóm 1 bước (User + Image + Event) | {nb['one_hop_G']:.2f} | {nb['one_hop_G_causal']:.2f} |",
          f"| Tweet khác trong 2 bước | {nb['two_hop_tweets_G']:.0f} | {nb['two_hop_tweets_G_causal']:.0f} |",
          "", "Tweet khác trong 2 bước, tách theo đường đi: "
          + ", ".join(f"{k} {v:.1f}" for k, v in two_hop.items()) + ".",
          f"\nSố chiều đặc trưng Tweet: {tweet_dim} (mpnet 768 + tweet_features 18 + CLIPcons 1).", ""]

    with open(os.path.join(DIAG_DIR, "step0_report.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(DIAG_DIR, "step0_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
