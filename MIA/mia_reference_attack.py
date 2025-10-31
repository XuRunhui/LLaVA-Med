#!/usr/bin/env python3
"""
Reference Non-Member Inference Attack
Uses non-member set as reference (known non-members) to detect members
"""

import json
import numpy as np
import argparse
import random
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support


def load_data(member_file, non_member_file, temperature, metric):
    """Load similarity scores"""
    with open(member_file, 'r') as f:
        member_data_all = json.load(f)
    with open(non_member_file, 'r') as f:
        non_member_data_all = json.load(f)

    member_data = [item[f'similarity_{temperature}'][metric] for item in member_data_all]
    non_member_data = [item[f'similarity_{temperature}'][metric] for item in non_member_data_all]

    return member_data, non_member_data


def reference_non_member_inference(member_data, non_member_data, granularity, both_non_member=False):
    """
    Reference attack using non-member set as reference.

    Algorithm:
    1. Split non-member set in half (reference vs target)
    2. For 1000 iterations:
       - Sample granularity samples from each set
       - Compare member target vs non-member reference (expect different)
       - Compare non-member target vs non-member reference (expect similar)
    3. Calculate AUC based on p-values

    Lower p-value means sample is significantly different from reference (likely member)

    Args:
        member_data: First dataset (members if both_non_member=False, else non-members)
        non_member_data: Second dataset (non-members, used as reference)
        granularity: Number of samples per test
        both_non_member: If True, treats both sets as non-members (e.g., SLAKE test vs PathVQA)
    """
    if both_non_member:
        # Both sets are non-members - we expect NO significant difference
        # Reference: non_member_data (e.g., PathVQA)
        # Target 1: member_data (e.g., SLAKE test - actually non-member)
        # Target 2: non_member_data (e.g., PathVQA - also non-member)
        random.shuffle(non_member_data)
        half = len(non_member_data) // 2
        reference_non_member = non_member_data[:half]
        target_non_member = non_member_data[half:]
        target_member = member_data  # Actually non-member, but kept for consistency
    else:
        # Original: member_data is training data, non_member_data is held-out
        random.shuffle(non_member_data)
        half = len(non_member_data) // 2
        reference_non_member = non_member_data[:half]
        target_non_member = non_member_data[half:]
        target_member = member_data

    # Adjust granularity if datasets are too small
    effective_granularity = min(
        granularity,
        len(reference_non_member),
        len(target_non_member),
        len(target_member)
    )

    if effective_granularity < granularity:
        print(f"WARNING: Granularity adjusted from {granularity} to {effective_granularity}")
        print(f"  Reference non-member: {len(reference_non_member)} samples")
        print(f"  Target non-member: {len(target_non_member)} samples")
        print(f"  Target member: {len(target_member)} samples")

    p_list = []
    label_list = []

    for _ in range(1000):
        samples_target_member = random.sample(target_member, effective_granularity)
        samples_reference_non_member = random.sample(reference_non_member, effective_granularity)
        samples_target_non_member = random.sample(target_non_member, effective_granularity)

        mean_target_member = np.mean(samples_target_member)
        mean_reference_non_member = np.mean(samples_reference_non_member)
        mean_target_non_member = np.mean(samples_target_non_member)

        var_target_member = np.var(samples_target_member, ddof=1)
        var_reference_non_member = np.var(samples_reference_non_member, ddof=1)
        var_target_non_member = np.var(samples_target_non_member, ddof=1)

        # Test member sample: compare against non-member reference
        # Expect members to have HIGHER similarity (higher score)
        denominator_member = np.sqrt(var_target_member / len(samples_target_member) +
                                     var_reference_non_member / len(samples_reference_non_member))
        if denominator_member == 0 or not np.isfinite(denominator_member):
            p_member = 0.5
        else:
            z_member = (mean_target_member - mean_reference_non_member) / denominator_member
            if not np.isfinite(z_member):
                p_member = 0.5
            else:
                p_member = 1 - norm.cdf(z_member)

        p_list.append(p_member)
        label_list.append(0)  # Label 0 for members (lower p-value expected)

        # Test non-member sample: compare against non-member reference
        # Expect non-members to have SIMILAR similarity
        denominator_non_member = np.sqrt(var_reference_non_member / len(samples_reference_non_member) +
                                         var_target_non_member / len(samples_target_non_member))
        if denominator_non_member == 0 or not np.isfinite(denominator_non_member):
            p_non_member = 0.5
        else:
            z_non_member = (mean_target_non_member - mean_reference_non_member) / denominator_non_member
            if not np.isfinite(z_non_member):
                p_non_member = 0.5
            else:
                p_non_member = 1 - norm.cdf(z_non_member)

        p_list.append(p_non_member)
        label_list.append(1)  # Label 1 for non-members (higher p-value expected)

    # Filter invalid values
    valid_indices = [i for i, p in enumerate(p_list) if np.isfinite(p)]
    if len(valid_indices) == 0:
        print("WARNING: All p-values are invalid.")
        return 0.5, 0.5, 0.5, 0.5, 0.5

    p_list_clean = [p_list[i] for i in valid_indices]
    label_list_clean = [label_list[i] for i in valid_indices]

    # Check for single class
    if len(set(label_list_clean)) < 2:
        print("WARNING: Only one class present.")
        return 0.5, 0.5, 0.5, 0.5, 0.5

    # Calculate AUC
    # Lower p-value = member, higher p-value = non-member
    auc = roc_auc_score(label_list_clean, p_list_clean)

    # Binary classification: threshold at 0.05
    # If p < 0.05, classify as member (label 0)
    pred_list = [0 if p < 0.05 else 1 for p in p_list_clean]
    accuracy = accuracy_score(label_list_clean, pred_list)
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_list_clean, pred_list, average='binary', pos_label=0, zero_division=0

    )

    return auc, accuracy, precision, recall, f1


def main(args):
    # Load data
    print("Loading similarity data...")
    member_data, non_member_data = load_data(
        args.member_similarity_file,
        args.non_member_similarity_file,
        args.temperature,
        args.similarity_metric
    )

    print(f"Member samples: {len(member_data)}")
    print(f"Non-member samples: {len(non_member_data)}")
    print(f"Granularity: {args.granularity}")
    print(f"Similarity metric: {args.similarity_metric}")

    # Run reference attack 5 times for stability
    print("\nRunning reference attack (5 trials)...")
    aucs = []
    accs = []
    precs = []
    recs = []
    f1s = []

    for trial in range(5):
        auc, acc, prec, rec, f1 = reference_non_member_inference(
            member_data, non_member_data, args.granularity, args.both_non_member
        )
        aucs.append(auc)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        print(f"Trial {trial + 1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

    # Calculate average and standard deviation
    avg_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    avg_prec = np.mean(precs)
    avg_rec = np.mean(recs)
    avg_f1 = np.mean(f1s)

    print(f"\n{'='*60}")
    print("REFERENCE ATTACK RESULTS")
    print(f"{'='*60}")
    print(f"AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Accuracy:  {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print(f"{'='*60}")
    print(f"\nInterpretation:")
    print(f"  Lower p-value → Classify as MEMBER")
    print(f"  Higher p-value → Classify as NON-MEMBER")
    if avg_auc > 0.7:
        print(f"  AUC = {avg_auc:.4f} → Strong privacy leakage detected!")
    elif avg_auc > 0.6:
        print(f"  AUC = {avg_auc:.4f} → Moderate privacy leakage")
    elif avg_auc > 0.55:
        print(f"  AUC = {avg_auc:.4f} → Weak privacy leakage")
    else:
        print(f"  AUC = {avg_auc:.4f} → No significant privacy leakage")

    # Save results
    results = {
        'attack_type': 'reference_non_member',
        'auc': float(avg_auc),
        'auc_std': float(std_auc),
        'accuracy': float(avg_acc),
        'accuracy_std': float(std_acc),
        'precision': float(avg_prec),
        'recall': float(avg_rec),
        'f1': float(avg_f1),
        'granularity': args.granularity,
        'temperature': args.temperature,
        'similarity_metric': args.similarity_metric,
        'n_members': len(member_data),
        'n_non_members': len(non_member_data),
        'trials': 5
    }

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reference Non-Member Inference Attack - Uses non-member set as reference'
    )
    parser.add_argument('--member-similarity-file', type=str, required=True,
                        help='JSON file with member similarity scores')
    parser.add_argument('--non-member-similarity-file', type=str, required=True,
                        help='JSON file with non-member similarity scores (used as reference)')
    parser.add_argument('--granularity', type=int, default=50,
                        help='Number of samples per statistical test (default: 50)')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature used for generation (default: 0.1)')
    parser.add_argument('--similarity-metric', type=str, default='rouge2_f',
                        choices=['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'bertscore_f'],
                        help='Similarity metric to use (default: rouge2_f)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file to save results (JSON)')
    parser.add_argument('--both-non-member', action='store_true',
                        help='If set, treats both datasets as non-members (e.g., SLAKE test vs PathVQA)')

    args = parser.parse_args()
    main(args)
