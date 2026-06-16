"""
03_clip_consistency.py
===========================
Tính feature NHẤT QUÁN tiêu đề-ảnh: cosine(CLIP-text(title), CLIP-image(thumbnail)).

Cơ sở: độ lệch ngữ nghĩa giữa văn bản và hình ảnh là tín hiệu tin giả kinh điển
(SAFE, Zhou et al. PAKDD 2020 — ref [12] trong paper). CLIP được huấn luyện để
text-image khớp nhau có cosine cao → bài đăng có tiêu đề không khớp ảnh
(misleading thumbnail, false connection) sẽ có cosine thấp.

Input : data/processed/posts_enriched.csv (thứ tự chuẩn), image_embeddings.npy
        (CLIP ViT-B/32, đã L2-normalize trong 01_prepare_data.py)
Output: data/processed/clip_cons.npy (N×1, float32)
        + copy sang data/processed_confounded/ (nội dung không đổi giữa 2 bộ)

Chạy:  PYTHONUTF8=1 uv run python 03_clip_consistency.py
"""
import os
import shutil
import sys
import numpy as np
import pandas as pd

SRC = os.path.join("data", "processed")
DST_CONF = os.path.join("data", "processed_confounded")
BATCH = 64

posts = pd.read_csv(os.path.join(SRC, "posts_enriched.csv"))
img_emb = np.load(os.path.join(SRC, "image_embeddings.npy"))
assert len(posts) == len(img_emb), "posts_enriched và image_embeddings lệch thứ tự/độ dài"
titles = posts["title"].fillna("").astype(str).tolist()
print(f"{len(titles)} bài | image_emb {img_emb.shape}")

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(titles), BATCH):
            batch = titles[i:i + BATCH]
            inputs = processor(text=batch, return_tensors="pt",
                               padding=True, truncation=True, max_length=77)
            # Dùng text_model + text_projection trực tiếp — tránh vấn đề
            # ModelOutput wrapping giữa các phiên bản transformers (như 01_prepare_data.py)
            out = model.text_model(input_ids=inputs["input_ids"],
                                   attention_mask=inputs["attention_mask"])
            t = model.text_projection(out.pooler_output)
            t = torch.nn.functional.normalize(t.float(), p=2, dim=-1)
            feats.append(t.numpy())
            if (i // BATCH) % 10 == 0:
                print(f"  text-encode {i}/{len(titles)}")
    text_emb = np.concatenate(feats, axis=0).astype(np.float32)
except Exception as e:
    print(f"[LỖI NGHIÊM TRỌNG] CLIP text encoder thất bại: {e}")
    sys.exit(1)

# image_embeddings đã chuẩn hóa L2 trong 01; ảnh lỗi = vector 0 → cosine = 0 (trung tính)
cos = np.sum(text_emb * img_emb, axis=1, keepdims=True).astype(np.float32)
np.save(os.path.join(SRC, "clip_cons.npy"), cos)
print(f"Đã lưu {SRC}/clip_cons.npy shape={cos.shape} | "
      f"mean={cos.mean():.3f} std={cos.std():.3f} | zero-img={(np.abs(cos) < 1e-6).sum()}")

if os.path.isdir(DST_CONF):
    shutil.copy(os.path.join(SRC, "clip_cons.npy"), os.path.join(DST_CONF, "clip_cons.npy"))
    print(f"Đã copy sang {DST_CONF}/")

# Sanity: cosine theo nhãn (chỉ in tham khảo, không dùng để tune)
lbl = posts["label_2way"].values
print(f"cos | Real(1): {cos[lbl == 1].mean():.4f} | Fake(0): {cos[lbl == 0].mean():.4f}")
