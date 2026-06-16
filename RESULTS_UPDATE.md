# RESULTS UPDATE — Số liệu chốt cho bản revision (2026-06-11)

Toàn bộ số dưới đây là **mean±std trên 3 seeds {42,1,2}**, sinh từ code trong repo,
trace được về file JSON trong `results/`. Đây là nguồn chuẩn duy nhất khi sửa docx.

## 1. Bảng chính (Table 1 mới của paper)

Nguồn: `results/final_tables.md` (sinh bởi `09_final_tables.py`).

| Model | Seen Acc% | Held-Out OOD% | Conf-Shift OOD% | AUC (conf) | F1 Drop% (conf) |
|---|---|---|---|---|---|
| MLP (content-only)  | 81.7±0.6 | 60.5±1.1 | 59.5±1.7 | 0.685 | 30.0 |
| ERM HeteroSAGE («Baseline GNN» mới) | 91.0±0.4 | 57.6±2.7 | 52.1±7.8 | 0.511 | 45.1 |
| IRM                 | 91.2±0.2 | 56.9±3.0 | 54.1±4.7 | 0.524 | 43.2 |
| EERM                | 93.2±0.2 | 60.9±1.9 | 59.6±1.1 | 0.694 | 38.6 |
| CausalHeteroGNN     | 83.8±0.5 | 57.7±0.8 | **74.2±3.6** | **0.851** | **12.7** |

Ghi chú quan trọng:
- «Baseline GNN» cũ (36.4%, AUC 0.314) là **nhánh baseline bên trong CausalHeteroGNN**
  (joint-trained, chung encoder) — KHÔNG dùng làm baseline chính nữa. Có thể giữ
  trong text như *diagnostic* (nhánh được khuyến khích hấp thụ spurious signal
  sụp đổ hoàn toàn — minh họa hiện tượng inversion), file: `metrics_bd_s*.json`.
- ERM độc lập: 52.1±7.8 (per-seed: 46.0 / 47.3 / 63.1), AUC 0.511 ≈ random.
  Câu chuyện đổi từ "decision boundary bị đảo" (chỉ đúng cho baseline-branch)
  thành "ERM mất hoàn toàn khả năng phân biệt (AUC ≈ ngẫu nhiên)".
- Held-Out OOD: số cũ trong paper (58.4 / 56.9) là single-seed — số đúng 3-seed
  là 57.2 (baseline-branch) / 57.7 (causal); với ERM mới là 57.6.
- Ordering conf-shift vẫn đứng vững với baseline trung thực:
  **ERM 52.1 < IRM 54.1 < MLP 59.5 ≈ EERM 59.6 < Causal 74.2**.

Nguồn file: `baselines_erm_mlp_{stdood,conf}_s{42,1,2}.json`,
`baselines_irm_eerm_{stdood,conf}_s*.json`, `metrics_{main,bd}_s*.json`.

## 2. Ablation domain label-history (MỚI — bảng/đoạn mới trong paper)

Nguồn: `metrics_nd_s{42,1,2}.json` (GNN_NEUTRAL_DOMAIN=1, conf-shift, transductive).

| Cấu hình | Conf-Shift OOD Acc% | OOD F1 | OOD AUC |
|---|---|---|---|
| CausalHeteroGNN (full)            | 74.2±3.6 | 0.731 | 0.851 |
| CausalHeteroGNN (domain fake-ratio = 0.5) | 58.8±2.9 | 0.534 | 0.681 |

**Diễn giải trung thực (bắt buộc đưa vào paper):** ~15.4 điểm trong 74.2% đến từ
feature lịch sử nguồn tin (Domain `fake_ratio_real` — target encoding từ label
train). Phần còn lại (~59%) trùng với trần content-only (MLP 59.5%). Kết luận
đúng: structural cut loại bỏ hoàn toàn confounder cộng đồng; hiệu năng tuyệt đối
còn dựa thêm vào *source credibility history* — một tín hiệu hợp lệ trong triển
khai nhưng sẽ kém bền nếu phân phối domain thay đổi. → Đây là câu trả lời chuẩn
bị sẵn cho phản biện, biến điểm yếu thành phân tích.

## 3. Worst-Group Accuracy (MỚI)

Nguồn: `results/worst_group_{stdood,conf}.json` (sinh bởi `08_worst_group.py`).

### Confounding-shift (nhóm = env × label, nhóm khó nhất = bị đảo tương quan)

| Model | Worst-Group Acc% | Avg-Group Acc% |
|---|---|---|
| ERM   | 23.5±7.9 | 73.2 |
| IRM   | 24.4±7.1 | 74.3 |
| EERM  | 28.6±1.3 | 77.0 |
| CausalHeteroGNN | **37.8±11.3** | 70.8 |

Mọi phương pháp soft-penalty sụp ở nhóm `spur_realbias|Fake` (fake post trong
env thiên-real) — đúng chữ ký shortcut. Causal là model duy nhất >33%.

### Standard OOD (nhóm = OOD subreddit)

Mọi model đều yếu ở nhóm `theonion|Fake` (nhận diện satire): 29–44%.
→ Limitation trung thực cần ghi: nhận diện satire bằng content là nhóm khó nhất,
chưa model nào giải quyết được; là hướng future work cụ thể.

## 4. Các số giữ nguyên (không đổi)

- FastRP leakage (§5.4): 93.6/94.3 → 61.0/61.1 — giữ nguyên.
- LFR (Table 2): giữ số, đổi framing — do(C₁) ≈ 0% là **sanity check by construction**,
  không phải finding; do(I)/do(D) là intervention thực sự có ý nghĩa.
- Graph statistics §3: 17,079 nodes / 28,274 edges — giữ.

## 5. Thay đổi code đã thực hiện

| File | Thay đổi |
|---|---|
| `07_baselines_erm_mlp.py` (mới) | ERM standalone + MLP content-only, schema tương thích |
| `08_worst_group.py` (mới) | Worst/avg-group acc từ checkpoint có sẵn, 2 protocol |
| `09_final_tables.py` (mới) | Gộp mọi kết quả → `results/final_tables.{md,json}` |
| `03_train_gnn.py` | + `GNN_NEUTRAL_DOMAIN` / `GNN_NEUTRAL_USER` (ablation); fix docstring dims (768/771); chú thích 2 định nghĩa inductive |
| `02_neo4j_import.py` | Fix CausalDAG metadata stale (ood_subreddits, grl_alpha=2.0, mô tả backdoor chính xác) |
| `01_prepare_data.py` | Comment 2-subreddit khớp config; random-fallback embedding → fail hard |

## 6. Checklist sửa docx (bước tiếp theo)

1. Abstract: 36.4→52.1 (ERM), thêm MLP anchor; "backdoor adjustment" → "structural intervention"; thêm 1 câu ablation domain.
2. §3: FastRP chỉ dùng cho §5.4 + BI, không nằm trong input mặc định.
3. §4: ngôn ngữ causal — structural intervention; ghi rõ Domain/User history KHÔNG bị cắt.
4. §5.1: nêu rõ inductive (held-out) vs transductive (conf-shift) + lý do.
5. §5.2: mô tả ERM standalone + MLP; baseline-branch chuyển thành diagnostic.
6. Table 1: thay số theo bảng mục 1 ở trên (sửa Held-Out: 57.6/56.9/60.9/57.7).
7. §5.3: viết lại đoạn kết quả chính theo ordering mới + worst-group.
8. §5.5: LFR do(C₁) → sanity check.
9. Thêm §5.6: domain-history ablation + content-only ceiling.
10. §6: cập nhật số + limitation satire + future work.
