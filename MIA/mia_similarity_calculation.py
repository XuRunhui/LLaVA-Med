#!/usr/bin/env python3
"""
Calculate similarity between generated responses and ground truth
For Reference and Target-Only attacks
"""

import json
import argparse
import torch
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore
from tqdm import tqdm
import numpy as np


def calculate_rouge(hypothesis, reference):
    """Calculate ROUGE scores"""
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
        # Return zeros if calculation fails
        return {key: 0.0 for key in [
            'rouge1_f', 'rouge1_p', 'rouge1_r',
            'rouge2_f', 'rouge2_p', 'rouge2_r',
            'rougeL_f', 'rougeL_p', 'rougeL_r'
        ]}


def calculate_bleu(hypothesis, reference):
    """Calculate BLEU score"""
    try:
        # Tokenize
        hyp_tokens = hypothesis.split()
        ref_tokens = [reference.split()]

        # Calculate BLEU with smoothing
        smooth = SmoothingFunction()
        bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth.method1)
        return bleu
    except:
        return 0.0


def calculate_bertscore(hypotheses, references, device='cuda', batch_size=128):
    """Calculate BERTScore (batched for efficiency, GPU-accelerated)"""
    try:
        # Use GPU if available, otherwise fall back to CPU
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        # Increase batch size for faster processing on GPU
        # A100 has 40GB+ memory, can handle larger batches
        P, R, F1 = bertscore(
            hypotheses,
            references,
            lang='en',
            verbose=False,
            device=device,
            batch_size=batch_size  # Larger batches = faster on GPU
            # num_layers=9  # Use fewer layers for speed (default is 12)
        )
        return P.tolist(), R.tolist(), F1.tolist()
    except:
        # Return zeros if calculation fails
        return [0.0] * len(hypotheses), [0.0] * len(hypotheses), [0.0] * len(hypotheses)


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

    # OPTIMIZATION 1: Pre-collect all data for BERTScore batching
    print("\n📊 Pre-processing data for batch BERTScore computation...")
    all_bertscore_data = {}  # {temperature: [(sample_idx, hypotheses, references)]}

    for temperature in args.temperatures:
        all_bertscore_data[temperature] = []

    similarity_results = []
    for sample_idx, sample in enumerate(tqdm(conversations, desc="Extracting data")):
        result = {'image_id': sample['image_id']}

        # Process each temperature
        for temperature in args.temperatures:
            temp_key = f"conversations_{temperature}"
            if temp_key not in sample:
                continue

            conv_list = sample[temp_key]

            # Extract generated responses and ground truth
            generated_responses = []
            ground_truth = None

            for conv_item in conv_list:
                if conv_item['from'].startswith('vlm'):
                    generated_responses.append(conv_item['value'])
                elif conv_item['from'] == 'ground truth':
                    ground_truth = conv_item['value']

            if not generated_responses or ground_truth is None:
                continue

            # Store for batch BERTScore calculation
            all_bertscore_data[temperature].append({
                'sample_idx': sample_idx,
                'hypotheses': generated_responses,
                'references': [ground_truth] * len(generated_responses)
            })

            # Calculate ROUGE and BLEU (fast, no GPU needed)
            similarities = []
            for gen_response in generated_responses:
                rouge_scores = calculate_rouge(gen_response, ground_truth)
                bleu_score = calculate_bleu(gen_response, ground_truth)
                similarities.append({
                    **rouge_scores,
                    'bleu': bleu_score
                })

            # Store partial results (will add BERTScore later)
            result[f'similarity_{temperature}'] = {
                'similarities': similarities,
                'count': len(similarities)
            }

        similarity_results.append(result)

    # OPTIMIZATION 2: Batch compute ALL BERTScores at once per temperature
    print("\n🚀 Computing BERTScore in large batches (GPU-accelerated)...")
    for temperature in args.temperatures:
        if not all_bertscore_data[temperature]:
            continue

        print(f"\n  Temperature {temperature}:")

        # Flatten all hypotheses and references
        all_hypotheses = []
        all_references = []
        sample_ranges = []  # Track which indices belong to which sample

        for data in all_bertscore_data[temperature]:
            start_idx = len(all_hypotheses)
            all_hypotheses.extend(data['hypotheses'])
            all_references.extend(data['references'])
            end_idx = len(all_hypotheses)
            sample_ranges.append({
                'sample_idx': data['sample_idx'],
                'start': start_idx,
                'end': end_idx
            })

        print(f"    Total pairs to compute: {len(all_hypotheses)}")

        # Compute ALL BERTScores at once
        if all_hypotheses:
            P_all, R_all, F1_all = calculate_bertscore(
                all_hypotheses,
                all_references,
                device=device
            )

            # Distribute results back to samples
            for range_info in sample_ranges:
                sample_idx = range_info['sample_idx']
                start = range_info['start']
                end = range_info['end']

                P_list = P_all[start:end]
                R_list = R_all[start:end]
                F1_list = F1_all[start:end]

                # Add BERTScore to the existing similarities
                result = similarity_results[sample_idx]
                temp_data = result[f'similarity_{temperature}']
                similarities = temp_data['similarities']

                for i, sim in enumerate(similarities):
                    sim['bertscore_p'] = P_list[i]
                    sim['bertscore_r'] = R_list[i]
                    sim['bertscore_f'] = F1_list[i]

                # Average all similarities
                if len(similarities) > 0:
                    avg_similarity = {}
                    for key in similarities[0].keys():
                        avg_similarity[key] = np.mean([s[key] for s in similarities])

                    result[f'similarity_{temperature}'] = avg_similarity

    # Save results
    print(f"\n💾 Saving similarity scores to: {args.similarity_json_path}")
    with open(args.similarity_json_path, 'w') as f:
        json.dump(similarity_results, f, indent=2)

    print(f"✅ Similarity calculation complete! Processed {len(similarity_results)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-json-path", type=str, required=True,
                        help="Path to conversation output file")
    parser.add_argument("--similarity-json-path", type=str, required=True,
                        help="Path to save similarity scores")
    parser.add_argument("--temperatures", nargs="+", type=float, required=True,
                        help="List of temperatures to process")

    args = parser.parse_args()
    main(args)
