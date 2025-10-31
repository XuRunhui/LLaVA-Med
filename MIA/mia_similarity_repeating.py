#!/usr/bin/env python3
"""
Calculate pairwise similarity between repeated generations
For Image-Only attack
"""

import json
import argparse
import torch
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
import numpy as np
from itertools import combinations
from tqdm import tqdm


def calculate_rouge(hypothesis, reference):
    """Calculate ROUGE scores between two texts"""
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return {
            'rouge1_f': scores['rouge-1']['f'],
            'rouge1_p': scores['rouge-1']['p'],
            'rouge1_r': scores['rouge-1']['r'],
            'rouge2_f': scores['rouge-2']['f'],
            'rouge2_p': scores['rouge-2']['p'],
            'rouge2_r': scores['rouge-2']['r'],
            'rougeL_f': scores['rouge-l']['f'],
            'rougeL_p': scores['rouge-l']['p'],
            'rougeL_r': scores['rouge-l']['r'],
        }
    except:
        # Return zeros if calculation fails (e.g., empty strings)
        return {key: 0.0 for key in [
            'rouge1_f', 'rouge1_p', 'rouge1_r',
            'rouge2_f', 'rouge2_p', 'rouge2_r',
            'rougeL_f', 'rougeL_p', 'rougeL_r'
        ]}


def calculate_bleu(hypothesis, reference):
    """Calculate BLEU score between two texts"""
    try:
        hyp_tokens = hypothesis.split()
        ref_tokens = [reference.split()]

        smooth = SmoothingFunction()
        bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth.method1)
        return bleu
    except:
        return 0.0


def calculate_bertscore_pairwise(text1, text2, device='cuda'):
    """Calculate BERTScore between two texts (GPU-accelerated)"""
    try:
        # Use GPU if available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        P, R, F1 = bertscore([text1], [text2], lang='en', verbose=False, device=device)
        return {
            'bertscore_p': P[0].item(),
            'bertscore_r': R[0].item(),
            'bertscore_f': F1[0].item()
        }
    except:
        return {
            'bertscore_p': 0.0,
            'bertscore_r': 0.0,
            'bertscore_f': 0.0
        }


def calculate_bertscore_batch(text_pairs, device='cuda'):
    """Calculate BERTScore for multiple pairs at once (more efficient)"""
    if not text_pairs:
        return []

    try:
        # Use GPU if available
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        # Unzip pairs
        texts1, texts2 = zip(*text_pairs)

        # Batch compute
        P, R, F1 = bertscore(list(texts1), list(texts2), lang='en', verbose=False, device=device, batch_size=64)

        # Return as list of dicts
        results = []
        for i in range(len(texts1)):
            results.append({
                'bertscore_p': P[i].item(),
                'bertscore_r': R[i].item(),
                'bertscore_f': F1[i].item()
            })
        return results
    except:
        # Return zeros if calculation fails
        return [{'bertscore_p': 0.0, 'bertscore_r': 0.0, 'bertscore_f': 0.0} for _ in text_pairs]


def main(args):
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected, using CPU (will be slower for BERTScore)")

    # Load conversations
    print(f"\nLoading conversations from: {args.conversation_json_path}")
    with open(args.conversation_json_path, 'r') as f:
        conversations = json.load(f)

    print(f"Total samples: {len(conversations)}")
    print(f"Repeating number: {args.repeating_num}")

    # Process each sample
    similarity_results = []

    for sample in tqdm(conversations, desc="Calculating pairwise similarity"):

        result = {'image_id': sample['image_id']}

        # Process each temperature
        for temperature in args.temperatures:
            temp_key = f"conversations_{temperature}"
            if temp_key not in sample:
                continue

            conv_list = sample[temp_key]

            # Extract repeated generated responses
            generated_responses = []
            for conv_item in conv_list:
                if conv_item['from'].startswith('vlm'):
                    generated_responses.append(conv_item['value'])

            # Check if we have enough repetitions
            if len(generated_responses) != args.repeating_num:
                print(f"Warning: Sample {sample['image_id']} has {len(generated_responses)} responses, expected {args.repeating_num}")
                continue

            # Calculate pairwise similarity between all pairs of repetitions
            pairwise_similarities = []

            # Get all pairs: (0,1), (0,2), ..., (n-2, n-1)
            pairs = list(combinations(range(len(generated_responses)), 2))

            # Calculate ROUGE and BLEU for all pairs first (fast operations)
            for i, j in pairs:
                text1 = generated_responses[i]
                text2 = generated_responses[j]

                # Calculate ROUGE
                rouge_scores = calculate_rouge(text1, text2)

                # Calculate BLEU
                bleu_score = calculate_bleu(text1, text2)

                # Store with pair indices
                pairwise_sim = {
                    **rouge_scores,
                    'bleu': bleu_score,
                    'pair': (i, j)
                }

                pairwise_similarities.append(pairwise_sim)

            # Calculate BERTScore for all pairs at once (GPU batch processing - much faster!)
            text_pairs = [(generated_responses[i], generated_responses[j]) for i, j in pairs]
            bert_scores_batch = calculate_bertscore_batch(text_pairs, device=device)

            # Add BERTScores to existing similarity results
            for idx, bert_scores in enumerate(bert_scores_batch):
                pairwise_similarities[idx].update(bert_scores)

            # Calculate average pairwise similarity
            if len(pairwise_similarities) > 0:
                avg_similarity = {}

                # Average all metrics except 'pair'
                metric_keys = [k for k in pairwise_similarities[0].keys() if k != 'pair']

                for key in metric_keys:
                    avg_similarity[key] = np.mean([s[key] for s in pairwise_similarities])

                # Also save min/max/std for analysis
                for key in metric_keys:
                    values = [s[key] for s in pairwise_similarities]
                    avg_similarity[f'{key}_std'] = np.std(values)
                    avg_similarity[f'{key}_min'] = np.min(values)
                    avg_similarity[f'{key}_max'] = np.max(values)

                # Save number of pairs
                avg_similarity['num_pairs'] = len(pairwise_similarities)

                result[f'similarity_{temperature}'] = avg_similarity

        similarity_results.append(result)

    # Save results
    print(f"\nSaving similarity scores to: {args.similarity_json_path}")
    with open(args.similarity_json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)

    print(f" Pairwise similarity calculation complete!")
    print(f"   Processed {len(similarity_results)} samples")
    print(f"   {len(pairs)} pairs per sample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate pairwise similarity for repeated generations (Image-Only attack)"
    )
    parser.add_argument("--conversation-json-path", type=str, required=True,
                        help="Path to conversation output file with repeated generations")
    parser.add_argument("--similarity-json-path", type=str, required=True,
                        help="Path to save pairwise similarity scores")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True,
                        help="List of temperatures to process (usually just one for image-only)")
    parser.add_argument("--repeating-num", type=int, required=True,
                        help="Number of repetitions per sample (e.g., 5)")

    args = parser.parse_args()

    # Validation
    if args.repeating_num < 2:
        raise ValueError("repeating-num must be at least 2 for pairwise similarity")

    main(args)
