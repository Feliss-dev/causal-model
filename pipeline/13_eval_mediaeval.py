"""
13_eval_mediaeval.py
====================
Cross-platform revalidation: chạy các checkpoint đã train trên Fakeddit (Reddit)
trên tập test MediaEval 2015 "Verifying Multimedia Use" (Twitter) — KHÔNG train lại.

Cách ánh xạ Twitter -> schema của model:
  - Mỗi tweet = 1 Post node mới, nối vào đồ thị Fakeddit KHÔNG có cạnh nào
    (= đúng protocol inductive content-only của 06_evaluate.py: test post bị cô lập).
  - Post feature = mpnet(clean tweet text) 768-d + 3 scalar Reddit
    [score, upvote_ratio, num_comments]. Twitter không có các trường này -> điền
    median của Fakeddit train rồi min-max theo thang Fakeddit (trung tính).
  - Mode "text_image" (phụ, không phải protocol của paper): thêm Image node
    (CLIP ViT-B/32 của ảnh đầu tiên) + cạnh has_image cho tweet có ảnh.
    Sự kiện Syrian_boy / Varoufakis là video -> không có ảnh.
  - Không dùng User/Domain/Subreddit (không có tương đương trên Twitter; fake_rate
    là đặc trưng dẫn xuất từ nhãn -> không được phép).

Nhãn: Fakeddit label_2way 0=Fake, 1=Real. MediaEval 'fake'->0, 'real'->1.
Metric chính của MediaEval 2015 là F1 lớp Fake; báo cáo thêm acc / macro-F1 / AUC
và tách theo event.

Cách chạy (từ project root):
  $env:PYTHONUTF8 = "1"
  uv run python pipeline/13_eval_mediaeval.py
Env:
  MEDIAEVAL_DIR  : mặc định D:/MediaTwitter/image-verification-corpus/mediaeval2015/testset
Output:
  data/mediaeval2015/*.npy (cache embedding)
  results/mediaeval2015_eval.json, results/mediaeval2015_eval.md,
  results/mediaeval2015_predictions.csv
"""

import os
import re
import sys
import json
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
train_gnn = importlib.import_module("05_train_gnn")
baselines6 = importlib.import_module("07_baselines_irm_eerm")
erm_mlp = importlib.import_module("08_baselines_erm_mlp")

MEDIAEVAL_DIR = os.environ.get(
    "MEDIAEVAL_DIR", "D:/MediaTwitter/image-verification-corpus/mediaeval2015/testset")
CACHE_DIR  = os.path.join("data", "mediaeval2015")
OUTPUT_DIR = "results"
MODEL_DIR  = "models"
SEEDS      = ["42", "1", "2"]
os.makedirs(CACHE_DIR, exist_ok=True)


# ===================== LOAD + CLEAN TWEETS =====================

def load_tweets():
    path = os.path.join(MEDIAEVAL_DIR, "tweets.txt")
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=3, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fake", "real"])].reset_index(drop=True)
    df["label_2way"] = (df["label"] == "real").astype(int)   # 0=Fake, 1=Real
    df["image_ids"] = df["imageId(s)"].str.strip()
    df["first_image"] = df["image_ids"].str.split(",").str[0].str.strip()
    # event = tiền tố của image id (eclipse_01 -> eclipse, syrianboy_1 -> syrianboy)
    df["event"] = df["first_image"].str.replace(r"_(\d+|video)$", "", regex=True)
    df["clean_text"] = df["tweetText"].apply(clean_tweet)
    return df


def clean_tweet(t):
    """Đưa tweet về dạng gần với Fakeddit clean_title (lowercase, bỏ URL/mention/ký tự)."""
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\bRT\b", " ", t)
    t = re.sub(r"@\w+:?", " ", t)
    t = t.replace("#", " ").replace("\\n", " ")
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ===================== EMBEDDINGS (cached) =====================

def text_embeddings(texts):
    path = os.path.join(CACHE_DIR, "text_mpnet.npy")
    if os.path.exists(path):
        emb = np.load(path)
        if len(emb) == len(texts):
            return emb
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    emb = model.encode(texts, show_progress_bar=True, batch_size=64,
                       convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    np.save(path, emb)
    return emb


def find_image_files():
    root = os.path.join(MEDIAEVAL_DIR, "MediaEval2015_TestSetImages", "TestSetImages")
    files = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
                files[stem] = os.path.join(dirpath, n)
    return files


def image_embeddings(image_ids):
    """CLIP ViT-B/32 image embeddings — giống extract_clip_embeddings của 01."""
    path = os.path.join(CACHE_DIR, "image_clip.npy")
    ids_path = os.path.join(CACHE_DIR, "image_ids.json")
    if os.path.exists(path) and os.path.exists(ids_path):
        cached_ids = json.load(open(ids_path))
        if cached_ids == image_ids:
            return np.load(path)
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    files = find_image_files()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    out = []
    for iid in image_ids:
        img = Image.open(files[iid]).convert("RGB")
        inputs = proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            v = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            out.append(model.visual_projection(v).numpy())
    emb = np.concatenate(out).astype(np.float32)
    np.save(path, emb)
    json.dump(image_ids, open(ids_path, "w"))
    return emb


# ===================== GRAPH AUGMENTATION =====================

def scalar_features(posts_df, n):
    """Twitter không có score/upvote_ratio/num_comments -> median Fakeddit train,
    chuẩn hoá theo đúng min/max mà build_heterodata đã dùng."""
    raw = posts_df[["score", "upvote_ratio", "num_comments"]].values.astype(np.float32)
    mn, mx = raw.min(axis=0), raw.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    med = posts_df.loc[posts_df["split"] == "train",
                       ["score", "upvote_ratio", "num_comments"]].median().values.astype(np.float32)
    return np.tile((med - mn) / rng, (n, 1)).astype(np.float32)


def append_tweets(data, posts_df, text_emb, tw, img_emb=None, img_ids=None):
    """Thêm tweet làm Post node cô lập (và tuỳ chọn Image node + cạnh has_image).
    Trả về (x_dict, edge_index_dict, tweet_post_idx)."""
    x_dict = {k: v.clone() for k, v in data.x_dict.items()}
    edge_dict = {k: v.clone() for k, v in data.edge_index_dict.items()}
    n_post0 = x_dict["Post"].shape[0]
    feats = np.concatenate([text_emb, scalar_features(posts_df, len(text_emb))], axis=1)
    assert feats.shape[1] == x_dict["Post"].shape[1], (feats.shape, x_dict["Post"].shape)
    x_dict["Post"] = torch.cat([x_dict["Post"], torch.tensor(feats)], dim=0)
    tweet_idx = torch.arange(n_post0, n_post0 + len(text_emb))

    if img_emb is not None:
        n_img0 = x_dict["Image"].shape[0]
        x_dict["Image"] = torch.cat([x_dict["Image"], torch.tensor(img_emb)], dim=0)
        img_pos = {iid: n_img0 + i for i, iid in enumerate(img_ids)}
        src, dst = [], []
        for j, iid in enumerate(tw["first_image"]):
            if iid in img_pos:
                src.append(n_post0 + j)
                dst.append(img_pos[iid])
        e = torch.tensor([src, dst], dtype=torch.long)
        k, rk = ("Post", "has_image", "Image"), ("Image", "rev_has_image", "Post")
        edge_dict[k] = torch.cat([edge_dict[k], e], dim=1)
        edge_dict[rk] = torch.cat([edge_dict[rk], e.flip(0)], dim=1)
    return x_dict, edge_dict, tweet_idx


# ===================== MODELS =====================

def load_models(data, subreddits_df, user_feat_dim, domain_feat_dim, post_feat_dim):
    """Trả về list (name, seed, fn(x_dict, edge_dict)->logits 2-way)."""
    models = []
    for s in SEEDS:
        ckpt = os.path.join(MODEL_DIR, f"causal_gnn_main_s{s}.pt")
        if os.path.exists(ckpt):
            m = train_gnn.CausalHeteroGNN(
                data.metadata(), hidden_channels=train_gnn.HIDDEN_CHANNELS,
                num_subreddits=len(subreddits_df), user_feat_dim=user_feat_dim,
                domain_feat_dim=domain_feat_dim, post_feat_dim=post_feat_dim,
                grl_alpha=train_gnn.GRL_ALPHA, edge_dropout=0.0,
                causal_cut=train_gnn.CAUSAL_CUT, autocut=train_gnn.AUTOCUT)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            models.append(("CausalHeteroGNN (causal head)", s, lambda x, e, m=m: m(x, e)[1]))
            models.append(("CausalHeteroGNN (baseline branch)", s, lambda x, e, m=m: m(x, e)[0]))
        ckpt = os.path.join(MODEL_DIR, f"erm_stdood_s{s}.pt")
        if os.path.exists(ckpt):
            m = baselines6.GNNClassifier(data.metadata(), train_gnn.HIDDEN_CHANNELS,
                                         train_gnn.DROPOUT, post_feat_dim,
                                         user_feat_dim, domain_feat_dim)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            models.append(("ERM (HeteroSAGE)", s, lambda x, e, m=m: m(x, e)))
        ckpt = os.path.join(MODEL_DIR, f"mlp_stdood_s{s}.pt")
        if os.path.exists(ckpt):
            m = erm_mlp.MLPContentOnly(post_feat_dim, train_gnn.HIDDEN_CHANNELS, train_gnn.DROPOUT)
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            m.eval()
            models.append(("MLP (content-only)", s, lambda x, e, m=m: m(x, e)))
    return models


# ===================== METRICS =====================

def metrics(y, logits):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
    p_fake = F.softmax(logits, dim=1)[:, 0].numpy()
    pred = logits.argmax(1).numpy()
    y = np.asarray(y)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_fake": float(f1_score(y, pred, pos_label=0, zero_division=0)),
        "pred_fake_rate": float((pred == 0).mean()),
        "confusion_matrix[Fake,Real]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    out["auc_fake"] = float(roc_auc_score(1 - y, p_fake)) if len(set(y)) == 2 else None
    return out, p_fake, pred


def agg(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


# ===================== MAIN =====================

def main():
    tw = load_tweets()
    y = tw["label_2way"].values
    print(f"MediaEval2015 test: {len(tw)} tweets | fake={int((y == 0).sum())} real={int((y == 1).sum())}")
    print(tw.groupby(["event", "label"]).size().unstack(fill_value=0))

    text_emb = text_embeddings(tw["clean_text"].tolist())
    files = find_image_files()
    img_ids = sorted({i for i in tw["first_image"] if i in files})
    img_emb = image_embeddings(img_ids)
    has_img = tw["first_image"].isin(img_ids).values
    print(f"Text emb {text_emb.shape} | images {img_emb.shape} | tweets with image: {has_img.sum()}")

    data, posts_df, subreddits_df, _, _, _, _, _, user_feat_dim, domain_feat_dim, post_feat_dim = \
        train_gnn.build_heterodata()
    models = load_models(data, subreddits_df, user_feat_dim, domain_feat_dim, post_feat_dim)
    print(f"Loaded {len(models)} model heads: {sorted({n for n, _, _ in models})}")

    modes = {
        "text_only": append_tweets(data, posts_df, text_emb, tw),
        "text_image": append_tweets(data, posts_df, text_emb, tw, img_emb, img_ids),
    }

    events = sorted(tw["event"].unique())
    runs, preds = [], pd.DataFrame({"tweetId": tw["tweetId"], "event": tw["event"], "label": tw["label"]})
    for mode, (x_dict, edge_dict, tidx) in modes.items():
        for name, seed, fn in models:
            if mode == "text_image" and name.startswith("MLP"):
                continue  # MLP không dùng graph -> giống hệt text_only
            with torch.no_grad():
                logits = fn(x_dict, edge_dict)[tidx]
            overall, p_fake, pred = metrics(y, logits)
            r = {"mode": mode, "model": name, "seed": seed, "overall": overall,
                 "with_image": metrics(y[has_img], logits[has_img])[0],
                 "by_event": {ev: metrics(y[(tw["event"] == ev).values],
                                          logits[(tw["event"] == ev).values])[0] for ev in events}}
            runs.append(r)
            preds[f"p_fake|{mode}|{name}|s{seed}"] = p_fake
            print(f"[{mode:<10}] {name:<34} s{seed:<3} acc={overall['accuracy']:.4f} "
                  f"F1fake={overall['f1_fake']:.4f} macroF1={overall['macro_f1']:.4f} "
                  f"AUC={overall['auc_fake']:.4f} predFake={overall['pred_fake_rate']:.2f}")

    # ---- aggregate mean±std over seeds ----
    summary = []
    for mode in modes:
        for name in dict.fromkeys(n for n, _, _ in models):
            rs = [r for r in runs if r["mode"] == mode and r["model"] == name]
            if not rs:
                continue
            row = {"mode": mode, "model": name, "n_seeds": len(rs)}
            for k in ["accuracy", "f1_fake", "macro_f1", "auc_fake", "pred_fake_rate"]:
                row[k] = agg([r["overall"][k] for r in rs])
                row["with_image_" + k] = agg([r["with_image"][k] for r in rs])
            row["event_acc"] = {ev: agg([r["by_event"][ev]["accuracy"] for r in rs]) for ev in events}
            row["event_auc"] = {ev: agg([r["by_event"][ev]["auc_fake"] for r in rs]) for ev in events}
            summary.append(row)

    out = {
        "_config": {
            "test_set": MEDIAEVAL_DIR, "n_tweets": int(len(tw)),
            "n_fake": int((y == 0).sum()), "n_real": int((y == 1).sum()),
            "n_with_image": int(has_img.sum()),
            "protocol": "Fakeddit-trained checkpoints, no retraining; tweets appended as isolated "
                        "Post nodes (inductive content-only). text_image adds CLIP Image node + "
                        "has_image edge. Reddit scalars imputed with Fakeddit train median.",
            "majority_baseline_acc": float(max((y == 0).mean(), (y == 1).mean())),
        },
        "summary": summary,
        "runs": runs,
    }
    with open(os.path.join(OUTPUT_DIR, "mediaeval2015_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    preds.to_csv(os.path.join(OUTPUT_DIR, "mediaeval2015_predictions.csv"), index=False)
    write_markdown(out, events)
    print("\nSaved results/mediaeval2015_eval.{json,md} + mediaeval2015_predictions.csv")


def fmt(t, pct=False):
    if t is None or t[0] is None:
        return "—"
    m, s = t
    return f"{m*100:.1f} ± {s*100:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


def write_markdown(out, events):
    c = out["_config"]
    L = ["# MediaEval 2015 cross-platform revalidation", "",
         f"Test set: `{c['test_set']}` — {c['n_tweets']} tweets "
         f"({c['n_fake']} fake / {c['n_real']} real; {c['n_with_image']} with an image file). "
         f"Majority-class accuracy = {c['majority_baseline_acc']*100:.1f}%.", "",
         c["protocol"], "", "Mean ± std over seeds 42/1/2. AUC uses P(Fake) with Fake as positive.", ""]
    for mode in ["text_only", "text_image"]:
        L += [f"## {mode}", "",
              "| Model | Acc % | F1 (Fake) % | Macro-F1 % | AUC | Pred. fake % |",
              "|---|---|---|---|---|---|"]
        for r in out["summary"]:
            if r["mode"] == mode:
                L.append(f"| {r['model']} | {fmt(r['accuracy'], 1)} | {fmt(r['f1_fake'], 1)} | "
                         f"{fmt(r['macro_f1'], 1)} | {fmt(r['auc_fake'])} | {fmt(r['pred_fake_rate'], 1)} |")
        L.append("")
    L += ["## Per-event accuracy % (text_only)", "",
          "| Model | " + " | ".join(events) + " |", "|---|" + "---|" * len(events)]
    for r in out["summary"]:
        if r["mode"] == "text_only":
            L.append(f"| {r['model']} | " + " | ".join(fmt(r["event_acc"][e], 1) for e in events) + " |")
    with open(os.path.join(OUTPUT_DIR, "mediaeval2015_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
