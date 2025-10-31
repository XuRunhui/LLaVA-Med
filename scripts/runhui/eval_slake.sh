#!/bin/bash
#SBATCH --job-name=slake_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

################################################################################
# Organized Evaluation Script for LLaVA-Med
#
# Folder structure:
#   evaluation/
#   ├── slake/
#   │   ├── epoch1/
#   │   ├── epoch5/
#   │   └── epoch10/
#   ├── vqa_rad/
#   │   └── epoch10/
#   └── pathvqa/
#       └── epoch10/
################################################################################

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Model checkpoint and epoch identifier
MODEL_NAME="/scratch1/runhuixu/outputs/slake_new/checkpoint-10"
EPOCH_NAME="step_120"  # Change this to: epoch1, epoch5, epoch10, etc.

# Dataset paths
SLAKE_DIR="/scratch1/runhuixu/datasets/SLAKE/Slake1.0"
VQA_RAD_DIR="/scratch1/runhuixu/datasets/vqa_rad"
PATHVQA_DIR="/scratch1/runhuixu/datasets/pvqa"

# Base output directory
BASE_OUTPUT_DIR="/scratch1/runhuixu/evaluation"

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

declare -A DATASETS=(
    ["slake"]="$SLAKE_DIR"
    # ["vqa_rad"]="$VQA_RAD_DIR"
    # ["pathvqa"]="$PATHVQA_DIR"
)

echo "========================================"
echo "LLaVA-Med Organized Evaluation"
echo "========================================"
echo "Model: ${MODEL_NAME}"
echo "Epoch: ${EPOCH_NAME}"
echo "Base Output: ${BASE_OUTPUT_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-0}"
echo ""
echo "Output structure:"
for dataset_name in "${!DATASETS[@]}"; do
    echo "  ${BASE_OUTPUT_DIR}/${dataset_name}/${EPOCH_NAME}/"
done
echo "========================================"
echo ""

# ============================================================================
# STEP 1: Generate Candidate Answer Files (if not exist)
# ============================================================================

echo "========================================="
echo "Step 1: Checking candidate answer files"
echo "========================================="
echo ""

for dataset_name in "${!DATASETS[@]}"; do
    dataset_dir="${DATASETS[$dataset_name]}"
    candidate_file="${dataset_dir}/train_open_answers.json"
    train_file="${dataset_dir}/llava_train.json"

    if [ ! -f "$candidate_file" ]; then
        echo "⚠️  Candidate file not found for ${dataset_name}"
        echo "   Generating from: ${train_file}"

        python -c "
import json

# Load training data
with open('${train_file}', 'r') as f:
    train_data = json.load(f)

# Extract unique open-ended answers
open_answers = set()
for item in train_data:
    if item.get('answer_type') == 'OPEN' or 'answer_type' not in item:
        if 'conversations' in item and len(item['conversations']) > 1:
            answer = item['conversations'][1]['value'].lower().strip()
            open_answers.add(answer)
        elif 'answer' in item:
            answer = item['answer'].lower().strip()
            open_answers.add(answer)

# Save candidate file
candidate_data = {'0': sorted(list(open_answers))}

with open('${candidate_file}', 'w') as f:
    json.dump(candidate_data, f, indent=2)

print(f'✅ Created candidate file with {len(open_answers)} unique answers')
"
    else
        num_candidates=$(python -c "import json; data=json.load(open('${candidate_file}')); print(len(data.get('0', [])))")
        echo "✅ Candidate file exists: ${dataset_name} (${num_candidates} unique answers)"
    fi
    echo ""
done

# ============================================================================
# STEP 2: Run Inference
# ============================================================================

echo "========================================="
echo "Step 2: Running inference"
echo "========================================="
echo ""

for dataset_name in "${!DATASETS[@]}"; do
    dataset_dir="${DATASETS[$dataset_name]}"

    # NEW: Organized output directory structure
    output_dir="${BASE_OUTPUT_DIR}/${dataset_name}/${EPOCH_NAME}"
    mkdir -p "${output_dir}"

    question_file="${dataset_dir}/llava_test.json"
    image_folder="${dataset_dir}/imgs"
    answers_file="${output_dir}/predictions.jsonl"

    echo "--- Processing ${dataset_name} (${EPOCH_NAME}) ---"
    echo "Question file: ${question_file}"
    echo "Image folder: ${image_folder}"
    echo "Output directory: ${output_dir}"
    echo "Predictions file: ${answers_file}"
    echo ""

    # Check if files exist
    if [ ! -f "${question_file}" ]; then
        echo "❌ Question file not found: ${question_file}"
        echo "   Skipping ${dataset_name}"
        echo ""
        continue
    fi

    if [ ! -d "${image_folder}" ]; then
        echo "❌ Image folder not found: ${image_folder}"
        echo "   Skipping ${dataset_name}"
        echo ""
        continue
    fi

    # Run inference
    echo "Starting inference..."
    python /scratch1/runhuixu/LLaVA-Med/llava/eval/downstream_inference.py \
        --model-name "${MODEL_NAME}" \
        --question-file "${question_file}" \
        --image-folder "${image_folder}" \
        --answers-file "${answers_file}" \
        --conv-mode "simple"

    echo "✅ Inference completed for ${dataset_name}"
    echo ""
done

# ============================================================================
# STEP 3: Evaluate Results
# ============================================================================

echo "========================================="
echo "Step 3: Evaluating results"
echo "========================================="
echo ""

for dataset_name in "${!DATASETS[@]}"; do
    dataset_dir="${DATASETS[$dataset_name]}"
    output_dir="${BASE_OUTPUT_DIR}/${dataset_name}/${EPOCH_NAME}"

    gt_file="${dataset_dir}/llava_test.json"
    candidate_file="${dataset_dir}/train_open_answers.json"
    pred_file="${output_dir}/predictions.jsonl"
    results_file="${output_dir}/metrics.txt"

    # Check if prediction file exists
    if [ ! -f "${pred_file}" ]; then
        echo "⚠️  Prediction file not found for ${dataset_name}, skipping evaluation"
        continue
    fi

    echo "--- Evaluating ${dataset_name} (${EPOCH_NAME}) ---"
    echo "Ground truth: ${gt_file}"
    echo "Candidate file: ${candidate_file}"
    echo "Predictions: ${pred_file}"
    echo "Results will be saved to: ${results_file}"
    echo ""

    python /scratch1/runhuixu/LLaVA-Med/llava/eval/run_eval.py \
        --gt "${gt_file}" \
        --candidate "${candidate_file}" \
        --pred "${pred_file}" \
        > "${results_file}"

    echo "--- ${dataset_name} (${EPOCH_NAME}) Results ---"
    cat "${results_file}"
    echo ""

    # Also save a summary file with metadata
    echo "Creating summary file..."
    cat > "${output_dir}/summary.txt" << EOF
======================================
Evaluation Summary
======================================
Dataset: ${dataset_name}
Epoch: ${EPOCH_NAME}
Model: ${MODEL_NAME}
Date: $(date)
======================================

Metrics:
$(cat "${results_file}")

Files:
- Predictions: predictions.jsonl
- Metrics: metrics.txt
- Ground Truth: ${gt_file}
- Candidate File: ${candidate_file}
======================================
EOF

    echo "✅ Summary saved to: ${output_dir}/summary.txt"
    echo ""
done

# ============================================================================
# STEP 4: Final Summary
# ============================================================================

echo "========================================="
echo "Evaluation Complete!"
echo "========================================="
echo ""
echo "Results directory structure:"
echo "${BASE_OUTPUT_DIR}/"
for dataset_name in "${!DATASETS[@]}"; do
    output_dir="${BASE_OUTPUT_DIR}/${dataset_name}/${EPOCH_NAME}"
    if [ -d "${output_dir}" ]; then
        echo "├── ${dataset_name}/"
        echo "│   └── ${EPOCH_NAME}/"
        echo "│       ├── predictions.jsonl    # Model predictions"
        echo "│       ├── metrics.txt          # Evaluation metrics"
        echo "│       └── summary.txt          # Complete summary"
    fi
done
echo ""
echo "To evaluate a different epoch:"
echo "  1. Update EPOCH_NAME variable (e.g., epoch5, epoch20)"
echo "  2. Update MODEL_NAME to point to that checkpoint"
echo "  3. Re-run this script"
echo ""
echo "Expected results (LLaVA-Med paper):"
echo "  - SLAKE: ~80.4% overall"
echo "  - VQA-RAD: ~61.6% overall"
echo "  - PathVQA: ~60.4% overall"
echo ""
