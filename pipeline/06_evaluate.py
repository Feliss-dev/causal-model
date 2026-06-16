import os
import json
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

# Dynamically import 05_train_gnn to avoid SyntaxError: invalid decimal literal
train_gnn = importlib.import_module("05_train_gnn")
CausalHeteroGNN = train_gnn.CausalHeteroGNN
build_heterodata = train_gnn.build_heterodata
compute_metrics = train_gnn.compute_metrics
compute_full_metrics = train_gnn.compute_full_metrics

# Paths
INPUT_DIR = os.environ.get("GNN_INPUT_DIR", os.path.join("data", "processed"))
OUTPUT_DIR = os.path.join("results")
MODEL_DIR = os.path.join("models")


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor,
                           t_range=(0.5, 10.0), n_steps=100) -> float:
    """
    Find the scalar temperature T that minimises NLL on a held-out set.
    Argmax (accuracy) is invariant to T; only probability calibration changes.
    Returns T as a Python float.
    """
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(t_range[0], t_range[1], n_steps):
        nll = F.cross_entropy(logits / t, labels).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t

def mask_post_edges(edge_index_dict, post_ids_to_mask: set):
    """
    Remove all edges incident on the given Post node indices.

    Simulates inductive (content-only) evaluation: masked nodes are classified
    from their own pre-computed features (text + CLIP + scalars) without
    aggregating neighbourhood context from the graph. This eliminates the
    transductive shortcut where a Post node in a pure-label subreddit trivially
    inherits its label via message-passing from its Subreddit/User neighbours.
    """
    masked = {}
    for edge_type, edge_idx in edge_index_dict.items():
        src_type, _, dst_type = edge_type
        if src_type == "Post":
            keep = torch.tensor(
                [s.item() not in post_ids_to_mask for s in edge_idx[0]],
                dtype=torch.bool,
            )
            masked[edge_type] = edge_idx[:, keep]
        elif dst_type == "Post":
            keep = torch.tensor(
                [d.item() not in post_ids_to_mask for d in edge_idx[1]],
                dtype=torch.bool,
            )
            masked[edge_type] = edge_idx[:, keep]
        else:
            masked[edge_type] = edge_idx
    return masked


def main():
    # Load dataset
    data, posts_df, subreddits_df, domains_df, post_map, sub_map, domain_map, img_map, \
        user_feat_dim, domain_feat_dim, post_feat_dim = build_heterodata()
    num_subreddits = len(subreddits_df)

    # Initialize Model — must match training config (hidden/grl/post dim)
    model = CausalHeteroGNN(data.metadata(),
                            hidden_channels=train_gnn.HIDDEN_CHANNELS,
                            num_subreddits=num_subreddits,
                            user_feat_dim=user_feat_dim, domain_feat_dim=domain_feat_dim,
                            post_feat_dim=post_feat_dim,
                            grl_alpha=train_gnn.GRL_ALPHA, edge_dropout=0.0,
                            causal_cut=train_gnn.CAUSAL_CUT,
                            autocut=train_gnn.AUTOCUT)

    run_tag = os.environ.get("GNN_RUN_TAG", "")
    skip_explain = os.environ.get("GNN_SKIP_EXPLAIN", "0") == "1"
    checkpoint_path = os.path.join(MODEL_DIR, f"causal_gnn{run_tag}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Please run 05_train_gnn.py first.")
        return

    print("Loading model checkpoint...")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # ── Temperature calibration on validation set ──────────────────────────────
    # Calibrate T to de-skew the softmax distribution (many logits push P→0/1).
    # T > 1 makes probabilities softer without changing accuracy (argmax-invariant).
    # Calibration is performed on the val set so test labels are never touched.
    val_mask = data['Post'].val_mask
    y_val_2way = data['Post'].y[val_mask]
    with torch.no_grad():
        (val_out_base_2, val_out_causal_2, _, _, _, _, _, _) = model(data.x_dict, data.edge_index_dict)
    T_base   = calibrate_temperature(val_out_base_2[val_mask],   y_val_2way)
    T_causal = calibrate_temperature(val_out_causal_2[val_mask], y_val_2way)
    print(f"Calibrated temperatures: T_baseline={T_base:.2f}, T_causal={T_causal:.2f}")

    test_mask = data['Post'].test_mask
    y_test_2way = data['Post'].y[test_mask]
    y_test_6way = data['Post'].y_6way[test_mask]

    # Seen/Unseen splits.
    # Prefer the explicit is_ood flag (ground-truth OOD marker, also used for the
    # confounding-shift benchmark where the spurious subreddit appears in train but
    # the OOD test reverses its correlation). Fall back to unseen-subreddit detection.
    test_posts = posts_df[posts_df["split"] == "test"].copy()
    if "is_ood" in test_posts.columns and test_posts["is_ood"].fillna(False).astype(bool).any():
        test_posts["is_seen_subreddit"] = ~test_posts["is_ood"].fillna(False).astype(bool)
    else:
        train_subs = set(posts_df[posts_df["split"] == "train"]["subreddit"])
        test_posts["is_seen_subreddit"] = test_posts["subreddit"].apply(lambda x: x in train_subs)

    test_indices = np.where(test_mask.numpy())[0]
    seen_test_indices = []
    unseen_test_indices = []

    for local_idx, row_idx in enumerate(test_indices):
        is_seen = test_posts.iloc[local_idx]["is_seen_subreddit"]
        if is_seen:
            seen_test_indices.append(row_idx)
        else:
            unseen_test_indices.append(row_idx)

    seen_test_mask = torch.tensor(seen_test_indices, dtype=torch.long)
    unseen_test_mask = torch.tensor(unseen_test_indices, dtype=torch.long)

    print(f"Evaluating splits: Total test={len(test_indices)} | Seen subreddits={len(seen_test_indices)} | Unseen subreddits={len(unseen_test_indices)}")

    # ── Full-graph (transductive) inference — used for overall metrics & OOD ──
    with torch.no_grad():
        (test_out_base_2, test_out_causal_2,
         test_out_base_6, test_out_causal_6, _, _, _, _) = model(data.x_dict, data.edge_index_dict)

    # ── Inductive inference for in-distribution seen-test ──────────────────────
    # Mask all Post edges for seen-test nodes so they cannot exploit the
    # subreddit-membership shortcut that inflates transductive accuracy to ~99%.
    # After masking, each Post is classified from its own node features only
    # (384-d text + 512-d CLIP + 3 scalar + 64-d FastRP), giving a realistic
    # content-based in-distribution accuracy.
    # Mask ALL test Post edges (seen + unseen) so the seen-vs-OOD comparison uses the
    # SAME inductive (content-only) inference mode → a methodologically fair F1-drop.
    print("Running inductive inference for ALL test nodes (seen + OOD, masking test Post edges)...")
    all_test_ids_set = set(int(i) for i in test_indices)
    inductive_edge_dict = mask_post_edges(data.edge_index_dict, all_test_ids_set)
    with torch.no_grad():
        (ind_out_base_2, ind_out_causal_2,
         ind_out_base_6, ind_out_causal_6, _, _, _, _) = model(data.x_dict, inductive_edge_dict)

    # Compute comprehensive metrics (just like in 06_train_gnn.py)
    label_names_2way = ["Fake", "Real"]
    label_names_6way = ["True", "Satire", "Misleading", "Imposter", "FalseConn", "Manipulated"]

    # Choose inference mode:
    #  - default (content-only): INDUCTIVE, all test Post edges masked -> leak-free OOD.
    #  - GNN_OOD_TRANSDUCTIVE=1: keep graph message-passing so the model DOES see the
    #    (reversed) subreddit confounder. Required for the confounding-shift benchmark,
    #    where the whole point is to test robustness to the spurious subreddit.
    transductive = os.environ.get("GNN_OOD_TRANSDUCTIVE", "0") == "1"
    if transductive:
        eb2, ec2, eb6, ec6 = test_out_base_2, test_out_causal_2, test_out_base_6, test_out_causal_6
        print("[eval mode] TRANSDUCTIVE (graph message-passing ON — confounder visible)")
    else:
        eb2, ec2, eb6, ec6 = ind_out_base_2, ind_out_causal_2, ind_out_base_6, ind_out_causal_6
        print("[eval mode] INDUCTIVE content-only (leak-free)")

    metrics_base_2way = compute_full_metrics(y_test_2way, eb2[test_mask], label_names_2way)
    metrics_causal_2way = compute_full_metrics(y_test_2way, ec2[test_mask], label_names_2way)
    metrics_base_6way = compute_full_metrics(y_test_6way, eb6[test_mask], label_names_6way)
    metrics_causal_6way = compute_full_metrics(y_test_6way, ec6[test_mask], label_names_6way)

    # OOD metrics
    ood_metrics = {}
    if len(seen_test_indices) > 0:
        y_test_seen_2way = data['Post'].y[seen_test_mask]
        ood_metrics["baseline_seen"] = compute_full_metrics(y_test_seen_2way, eb2[seen_test_mask], label_names_2way)
        ood_metrics["causal_seen"] = compute_full_metrics(y_test_seen_2way, ec2[seen_test_mask], label_names_2way)

    if len(unseen_test_indices) > 0:
        y_test_unseen_2way = data['Post'].y[unseen_test_mask]
        ood_metrics["baseline_unseen"] = compute_full_metrics(y_test_unseen_2way, eb2[unseen_test_mask], label_names_2way)
        ood_metrics["causal_unseen"] = compute_full_metrics(y_test_unseen_2way, ec2[unseen_test_mask], label_names_2way)

    # F1 Drop calculation
    f1_drop_base = 0.0
    f1_drop_causal = 0.0
    if "baseline_seen" in ood_metrics and "baseline_unseen" in ood_metrics:
        f1_seen_b = ood_metrics["baseline_seen"]["macro_f1"]
        f1_unseen_b = ood_metrics["baseline_unseen"]["macro_f1"]
        f1_drop_base = ((f1_seen_b - f1_unseen_b) / f1_seen_b * 100) if f1_seen_b > 0 else 0
        
        f1_seen_c = ood_metrics["causal_seen"]["macro_f1"]
        f1_unseen_c = ood_metrics["causal_unseen"]["macro_f1"]
        f1_drop_causal = ((f1_seen_c - f1_unseen_c) / f1_seen_c * 100) if f1_seen_c > 0 else 0

    metrics_res = {
        "baseline": {
            "overall_2way": metrics_base_2way,
            "overall_6way": metrics_base_6way,
            "seen_2way": ood_metrics.get("baseline_seen", {}),
            "unseen_2way": ood_metrics.get("baseline_unseen", {}),
            "f1_drop_pct": f1_drop_base,
            "overall": {
                "accuracy": metrics_base_2way["accuracy"],
                "f1": metrics_base_2way["macro_f1"],
                "auc": metrics_base_2way.get("auc_roc", 0.5),
                "f1_6way": metrics_base_6way["macro_f1"]
            },
            "seen": {
                "accuracy": ood_metrics.get("baseline_seen", {}).get("accuracy", 0),
                "f1": ood_metrics.get("baseline_seen", {}).get("macro_f1", 0),
                "auc": ood_metrics.get("baseline_seen", {}).get("auc_roc", 0),
                "f1_6way": 0
            },
            "unseen": {
                "accuracy": ood_metrics.get("baseline_unseen", {}).get("accuracy", 0),
                "f1": ood_metrics.get("baseline_unseen", {}).get("macro_f1", 0),
                "auc": ood_metrics.get("baseline_unseen", {}).get("auc_roc", 0),
                "f1_6way": 0
            },
            "f1_6way_drop_pct": 0
        },
        "causal": {
            "overall_2way": metrics_causal_2way,
            "overall_6way": metrics_causal_6way,
            "seen_2way": ood_metrics.get("causal_seen", {}),
            "unseen_2way": ood_metrics.get("causal_unseen", {}),
            "f1_drop_pct": f1_drop_causal,
            "overall": {
                "accuracy": metrics_causal_2way["accuracy"],
                "f1": metrics_causal_2way["macro_f1"],
                "auc": metrics_causal_2way.get("auc_roc", 0.5),
                "f1_6way": metrics_causal_6way["macro_f1"]
            },
            "seen": {
                "accuracy": ood_metrics.get("causal_seen", {}).get("accuracy", 0),
                "f1": ood_metrics.get("causal_seen", {}).get("macro_f1", 0),
                "auc": ood_metrics.get("causal_seen", {}).get("auc_roc", 0),
                "f1_6way": 0
            },
            "unseen": {
                "accuracy": ood_metrics.get("causal_unseen", {}).get("accuracy", 0),
                "f1": ood_metrics.get("causal_unseen", {}).get("macro_f1", 0),
                "auc": ood_metrics.get("causal_unseen", {}).get("auc_roc", 0),
                "f1_6way": 0
            },
            "f1_6way_drop_pct": 0
        }
    }

    if train_gnn.AUTOCUT:
        gates_final = torch.sigmoid(model.gate_logits).detach().cpu()
        metrics_res["learned_gates"] = {
            "__".join(et): round(float(g), 4)
            for et, g in zip(model.edge_types, gates_final)
        }
        print("Learned gates:", sorted(metrics_res["learned_gates"].items(),
                                       key=lambda x: x[1])[:4], "...")
    with open(os.path.join(OUTPUT_DIR, f"metrics{run_tag}.json"), "w") as f:
        json.dump(metrics_res, f, indent=4, default=str)
    print(f"Saved metrics{run_tag}.json successfully!")

    if skip_explain:
        print("[SKIP_EXPLAIN=1] Skipping counterfactual + causal-path. Done.")
        return

    # ------------------ COUNTERFACTUAL ENGINE ------------------
    print("\nGenerating Counterfactual interventions...")
    cf_results = []
    
    # Identify low-fake-rate subreddit (neutral) and domain (credible) for do-calculus swaps
    neutral_sub_id = subreddits_df.loc[subreddits_df["fake_ratio_real"].idxmin(), "sub_id"]
    neutral_sub_idx = sub_map[str(neutral_sub_id)]

    credible_domain_id = domains_df.loc[domains_df["fake_ratio_real"].idxmin(), "domain_id"]
    credible_domain_idx = domain_map[str(credible_domain_id)]

    original_edge_dict = {k: v.clone() for k, v in data.edge_index_dict.items()}

    for local_idx, row_idx in enumerate(test_indices):
        post_row = test_posts.iloc[local_idx]
        post_id = str(post_row["post_id"])
        post_pyg_idx = int(row_idx)

        # Temperature-scaled softmax for calibrated probabilities
        p_orig_base   = F.softmax(test_out_base_2[post_pyg_idx]   / T_base,   dim=0)[1].item()
        p_orig_causal = F.softmax(test_out_causal_2[post_pyg_idx] / T_causal, dim=0)[1].item()

        # CF1: Remove Image (do(has_image=0))
        cf_edge_dict = {k: v.clone() for k, v in original_edge_dict.items()}
        p_img_edge = cf_edge_dict[('Post', 'has_image', 'Image')]
        mask = p_img_edge[0] != post_pyg_idx
        cf_edge_dict[('Post', 'has_image', 'Image')] = p_img_edge[:, mask]

        rev_key = ('Image', 'rev_has_image', 'Post')
        if rev_key in cf_edge_dict:
            img_p_edge = cf_edge_dict[rev_key]
            mask_rev = img_p_edge[1] != post_pyg_idx
            cf_edge_dict[rev_key] = img_p_edge[:, mask_rev]

        with torch.no_grad():
            cf_out_base, cf_out_causal, _, _, _, _, _, _ = model(data.x_dict, cf_edge_dict)
        p_cf_img_base   = F.softmax(cf_out_base[post_pyg_idx]   / T_base,   dim=0)[1].item()
        p_cf_img_causal = F.softmax(cf_out_causal[post_pyg_idx] / T_causal, dim=0)[1].item()

        # CF2: Swap to Credible Domain (do(domain=credible))
        cf_edge_dict = {k: v.clone() for k, v in original_edge_dict.items()}
        p_dom_edge = cf_edge_dict[('Post', 'links_to', 'Domain')].clone()
        p_dom_edge[1, p_dom_edge[0] == post_pyg_idx] = credible_domain_idx
        cf_edge_dict[('Post', 'links_to', 'Domain')] = p_dom_edge
        
        rev_key = ('Domain', 'rev_links_to', 'Post')
        if rev_key in cf_edge_dict:
            dom_p_edge = cf_edge_dict[rev_key].clone()
            dom_p_edge[0, dom_p_edge[1] == post_pyg_idx] = credible_domain_idx
            cf_edge_dict[rev_key] = dom_p_edge

        with torch.no_grad():
            cf_out_base, cf_out_causal, _, _, _, _, _, _ = model(data.x_dict, cf_edge_dict)
        p_cf_dom_base   = F.softmax(cf_out_base[post_pyg_idx]   / T_base,   dim=0)[1].item()
        p_cf_dom_causal = F.softmax(cf_out_causal[post_pyg_idx] / T_causal, dim=0)[1].item()

        # CF3: Swap Subreddit to neutral (do(subreddit=neutral))
        cf_edge_dict = {k: v.clone() for k, v in original_edge_dict.items()}
        p_sub_edge = cf_edge_dict[('Post', 'posted_in', 'Subreddit')].clone()
        p_sub_edge[1, p_sub_edge[0] == post_pyg_idx] = neutral_sub_idx
        cf_edge_dict[('Post', 'posted_in', 'Subreddit')] = p_sub_edge

        rev_key = ('Subreddit', 'rev_posted_in', 'Post')
        if rev_key in cf_edge_dict:
            sub_p_edge = cf_edge_dict[rev_key].clone()
            sub_p_edge[0, sub_p_edge[1] == post_pyg_idx] = neutral_sub_idx
            cf_edge_dict[rev_key] = sub_p_edge

        with torch.no_grad():
            cf_out_base, cf_out_causal, _, _, _, _, _, _ = model(data.x_dict, cf_edge_dict)
        p_cf_sub_base   = F.softmax(cf_out_base[post_pyg_idx]   / T_base,   dim=0)[1].item()
        p_cf_sub_causal = F.softmax(cf_out_causal[post_pyg_idx] / T_causal, dim=0)[1].item()

        cf_results.append({
            "post_id": post_id,
            "title": post_row["title"],
            "subreddit": post_row["subreddit"],
            "domain": post_row["domain"],
            "label_true": int(post_row["label_2way"]),
            "original": {"baseline": p_orig_base, "causal": p_orig_causal},
            "cf_image": {"baseline": p_cf_img_base, "causal": p_cf_img_causal},
            "cf_domain": {"baseline": p_cf_dom_base, "causal": p_cf_dom_causal},
            "cf_subreddit": {"baseline": p_cf_sub_base, "causal": p_cf_sub_causal}
        })

    with open(os.path.join(OUTPUT_DIR, "counterfactuals.json"), "w") as f:
        json.dump(cf_results, f, indent=4)
    print("Saved counterfactuals.json successfully!")

    # ------------------ CAUSAL PATH EXPLAINABILITY ------------------
    print("\nCalculating Causal Path explanations using Gradient-based Attribution...")
    causal_paths = {}

    data['Subreddit'].x.requires_grad_(True)
    data['Domain'].x.requires_grad_(True)
    data['Image'].x.requires_grad_(True)

    # Temporary evaluation with gradient collection enabled
    # We turn on training mode to trace gradients of parameters
    model.train()
    
    # Forward run
    (pred_base_2, pred_causal_2, _, _, _, _, _, _) = model(data.x_dict, data.edge_index_dict)
    
    for local_idx, row_idx in enumerate(test_indices):
        post_row = test_posts.iloc[local_idx]
        post_id = str(post_row["post_id"])
        post_pyg_idx = int(row_idx)

        prob_fake = F.softmax(pred_causal_2[post_pyg_idx], dim=0)[1]

        model.zero_grad()
        if data['Subreddit'].x.grad is not None:
            data['Subreddit'].x.grad.zero_()
        if data['Domain'].x.grad is not None:
            data['Domain'].x.grad.zero_()
        if data['Image'].x.grad is not None:
            data['Image'].x.grad.zero_()

        prob_fake.backward(retain_graph=True)

        sub_name = post_row["subreddit"]
        sub_idx = sub_map.get(f"sub_{sub_name}", 0)
        
        dom_name = post_row["domain"]
        dom_idx = domain_map.get(f"domain_{dom_name}", 0)
        
        img_idx = img_map.get(f"img_{post_id}", 0)

        grad_sub = data['Subreddit'].x.grad[sub_idx].abs().sum().item() if data['Subreddit'].x.grad is not None else 0.0
        grad_dom = data['Domain'].x.grad[dom_idx].abs().sum().item() if data['Domain'].x.grad is not None else 0.0
        grad_img = data['Image'].x.grad[img_idx].abs().sum().item() if data['Image'].x.grad is not None else 0.0

        total_grad = grad_sub + grad_dom + grad_img + 1e-8
        
        attr_sub = grad_sub / total_grad
        attr_dom = grad_dom / total_grad
        attr_img = grad_img / total_grad

        attr_sub = max(0.0, min(1.0, attr_sub))
        attr_dom = max(0.0, min(1.0, attr_dom))
        attr_img = max(0.0, min(1.0, attr_img))
        
        sub_fake_rate_row = subreddits_df.loc[subreddits_df["name"] == sub_name, "fake_ratio_real"]
        sub_bias = float(sub_fake_rate_row.values[0]) if not sub_fake_rate_row.empty else 0.5
        dom_fake_rate_row = domains_df.loc[domains_df["url_domain"] == dom_name, "fake_ratio_real"]
        dom_cred = 1.0 - float(dom_fake_rate_row.values[0]) if not dom_fake_rate_row.empty else 0.5

        text_ratio = 0.25  # Default fallback

        if attr_dom > attr_sub and attr_dom > attr_img:
            explanation = f"Flagged primarily due to source credibility. The domain '{dom_name}' has low credibility (fake_ratio={1-dom_cred:.2f}) and represents {attr_dom:.1%} of the decision influence."
        elif attr_img > attr_sub and attr_img > attr_dom:
            explanation = f"Flagged due to visual factors. The image associated with the post represents {attr_img:.1%} of the decision influence, indicating a strong correlation with typical fake news layouts."
        else:
            explanation = f"Flagged due to subreddit platform context. The subreddit 'r/{sub_name}' has a historical fake rate of {sub_bias:.2f} and contributes {attr_sub:.1%} to the model's prediction."

        causal_paths[post_id] = {
            "post_id": post_id,
            "title": post_row["title"],
            "subreddit": {
                "name": sub_name,
                "bias": sub_bias,
                "attribution": attr_sub
            },
            "domain": {
                "name": dom_name,
                "credibility": dom_cred,
                "attribution": attr_dom
            },
            "image": {
                "attribution": attr_img,
                "text_ratio": text_ratio
            },
            "confidence": prob_fake.item(),
            "explanation": explanation
        }

    with open(os.path.join(OUTPUT_DIR, "causal_paths.json"), "w") as f:
        json.dump(causal_paths, f, indent=4)
    print("Saved causal_paths.json successfully!")
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
