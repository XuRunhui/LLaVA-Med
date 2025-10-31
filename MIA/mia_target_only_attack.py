#!/usr/bin/env python3
"""
Target-Only Member Inference Attack
Uses temperature difference to infer membership
Members show less temperature sensitivity than non-members
"""

import json
import numpy as np
import argparse
import random
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


def load_data(member_file, non_member_file, temp_low, temp_high, metric):
    """Load similarity scores for two temperatures"""
    with open(member_file, 'r') as f:
        member_data_all = json.load(f)
    with open(non_member_file, 'r') as f:
        non_member_data_all = json.load(f)

    # Extract similarity scores for both temperatures
    member_low = [item[f'similarity_{temp_low}'][metric] for item in member_data_all]
    member_high = [item[f'similarity_{temp_high}'][metric] for item in member_data_all]

    non_member_low = [item[f'similarity_{temp_low}'][metric] for item in non_member_data_all]
    non_member_high = [item[f'similarity_{temp_high}'][metric] for item in non_member_data_all]

    # Calculate temperature difference (low - high)
    # Members should have smaller differences (more stable)
    member_diff = [low - high for low, high in zip(member_low, member_high)]
    non_member_diff = [low - high for low, high in zip(non_member_low, non_member_high)]

    return member_diff, non_member_diff


def target_only_member_inference(member_data, non_member_data, granularity):
    """
    Target-Only attack: uses temperature difference as membership signal
    """
    p_list = []
    label_list = []

    # Run attack multiple times for stability
    for _ in range(1000):
        # Sample subsets
        samples_member = random.sample(member_data, min(granularity, len(member_data)))
        samples_non_member = random.sample(non_member_data, min(granularity, len(non_member_data)))

        # Calculate statistics
        mean_mem = np.mean(samples_member)
        var_mem = np.var(samples_member, ddof=1)

        mean_non_mem = np.mean(samples_non_member)
        var_non_mem = np.var(samples_non_member, ddof=1)

        # Z-test: members should have smaller temperature difference (with NaN/inf handling)
        denominator = np.sqrt(var_mem / len(samples_member) + var_non_mem / len(samples_non_member))
        if denominator == 0 or not np.isfinite(denominator):
            p_member = 0.5
        else:
            z_member = (mean_non_mem - mean_mem) / denominator
            if not np.isfinite(z_member):
                p_member = 0.5
            else:
                p_member = 1 - norm.cdf(z_member)

        p_list.append(p_member)
        label_list.append(1)  # Member

        # Z-test: non-members should have larger temperature difference (with NaN/inf handling)
        if denominator == 0 or not np.isfinite(denominator):
            p_non_member = 0.5
        else:
            z_non_member = (mean_mem - mean_non_mem) / denominator
            if not np.isfinite(z_non_member):
                p_non_member = 0.5
            else:
                p_non_member = 1 - norm.cdf(z_non_member)

        p_list.append(p_non_member)
        label_list.append(0)  # Non-member

    # Filter out any remaining NaN/inf values
    valid_indices = [i for i, p in enumerate(p_list) if np.isfinite(p)]
    if len(valid_indices) == 0:
        print("WARNING: All p-values are invalid. Temperature differences may be constant.")
        return 0.5, 0.5, 0.5, 0.5, 0.5

    p_list_clean = [p_list[i] for i in valid_indices]
    label_list_clean = [label_list[i] for i in valid_indices]

    # Calculate metrics
    if len(set(label_list_clean)) < 2:
        print("WARNING: Only one class in cleaned data.")
        return 0.5, 0.5, 0.5, 0.5, 0.5

    auc = roc_auc_score(label_list_clean, p_list_clean)

    # Convert to binary predictions (threshold = 0.5)
    pred_list = [1 if p < 0.5 else 0 for p in p_list_clean]
    accuracy = accuracy_score(label_list_clean, pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list_clean, pred_list, average='binary', zero_division=0)

    return auc, accuracy, precision, recall, f1


def main(args):
    # Load data
    print("Loading similarity data...")
    member_data, non_member_data = load_data(
        args.member_similarity_file,
        args.non_member_similarity_file,
        args.temperature_low,
        args.temperature_high,
        args.similarity_metric
    )

    print(f"Member samples: {len(member_data)}")
    print(f"Non-member samples: {len(non_member_data)}")
    print(f"Granularity: {args.granularity}")
    print(f"Temperature low: {args.temperature_low}")
    print(f"Temperature high: {args.temperature_high}")
    print(f"Similarity metric: {args.similarity_metric}")

    # Run attack multiple times
    print("\nRunning target-only attack (5 trials)...")
    aucs, accs, precs, recs, f1s = [], [], [], [], []

    for trial in range(5):
        auc, acc, prec, rec, f1 = target_only_member_inference(
            member_data, non_member_data, args.granularity
        )
        aucs.append(auc)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f"Trial {trial + 1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Average results
    avg_auc = np.mean(aucs)
    avg_acc = np.mean(accs)
    avg_prec = np.mean(precs)
    avg_rec = np.mean(recs)
    avg_f1 = np.mean(f1s)

    print(f"\n{'='*60}")
    print("TARGET-ONLY ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"AUC:       {avg_auc:.4f} ± {np.std(aucs):.4f}")
    print(f"Accuracy:  {avg_acc:.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {avg_prec:.4f} ± {np.std(precs):.4f}")
    print(f"Recall:    {avg_rec:.4f} ± {np.std(recs):.4f}")
    print(f"F1 Score:  {avg_f1:.4f} ± {np.std(f1s):.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        'attack_type': 'target_only',
        'auc': float(avg_auc),
        'auc_std': float(np.std(aucs)),
        'accuracy': float(avg_acc),
        'accuracy_std': float(np.std(accs)),
        'precision': float(avg_prec),
        'recall': float(avg_rec),
        'f1': float(avg_f1),
        'granularity': args.granularity,
        'temperature_low': args.temperature_low,
        'temperature_high': args.temperature_high,
        'similarity_metric': args.similarity_metric,
        'trials': aucs
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--member-similarity-file', type=str, required=True)
    parser.add_argument('--non-member-similarity-file', type=str, required=True)
    parser.add_argument('--granularity', type=int, default=50)
    parser.add_argument('--temperature-low', type=float, default=0.1)
    parser.add_argument('--temperature-high', type=float, default=1.5)
    parser.add_argument('--similarity-metric', type=str, default='rouge2_f',
                        choices=['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f'])
    parser.add_argument('--output-file', type=str, default=None)

    args = parser.parse_args()
    main(args)
