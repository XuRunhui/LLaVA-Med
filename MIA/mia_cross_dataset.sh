#!/bin/bash
#SBATCH --job-name=ref_attack_cross
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

################################################################################
# Reference Attack Only - Cross-Dataset (SLAKE vs PathVQA)
#
# This script runs ONLY the reference attack using:
# - Member set: SLAKE training data (model was trained on this)
# - Non-member set: PathVQA data (model never saw this)
#
# Steps:
# 1. Generate conversations for both datasets
# 2. Calculate similarity scores
# 3. Run reference attack
################################################################################

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

echo "=========================================="
echo "Reference Attack Only - Cross-Dataset"
echo "=========================================="
echo ""

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Model checkpoint (your fine-tuned model)
MODEL_PATH="/scratch1/runhuixu/checkpoints/slake/checkpoint-100"

# MEMBER SET (training data - SLAKE)
# MEMBER_DATA="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_train.json"
# MEMBER_IMAGE_FOLDER="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/imgs"
MEMBER_DATA="/scratch1/runhuixu/datasets/vqa_rad/train.json"
MEMBER_IMAGE_FOLDER="/scratch1/runhuixu/datasets/vqa_rad/VQA_RAD_Image_Folder"
# NON-MEMBER SET (different dataset - PathVQA)
NON_MEMBER_DATA="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/annotations/llava_train.json"
NON_MEMBER_IMAGE_FOLDER="/project2/ruishanl_1185/SDP_for_VLM/datasets/PathVQA/images"
# NON_MEMBER_DATA="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_test.json"
# NON_MEMBER_IMAGE_FOLDER="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/imgs"


# Output directory
# OUTPUT_DIR="/scratch1/runhuixu/mia_results/slake_train_vs_test"
OUTPUT_DIR="/scratch1/runhuixu/mia_results/radvqa_vs_pathvqa"
mkdir -p "$OUTPUT_DIR"

# Attack parameters
GRANULARITY=50  # Number of samples per statistical test
SIMILARITY_METRIC="rouge2_f"  # Options: rouge1_f, rouge2_f, rougeL_f, bleu, bertscore_f
CONV_MODE="simple"

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo ""
echo "  MEMBER SET (model was trained on this):"
echo "    Data: ${MEMBER_DATA}"
echo "    Images: ${MEMBER_IMAGE_FOLDER}"
echo ""
echo "  NON-MEMBER SET (model never saw this):"
echo "    Data: ${NON_MEMBER_DATA}"
echo "    Images: ${NON_MEMBER_IMAGE_FOLDER}"
echo ""
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Similarity metric: ${SIMILARITY_METRIC}"
echo "  Granularity: ${GRANULARITY}"
echo ""

# ============================================================================
# STEP 1: Prepare Member/Non-Member Sets
# ============================================================================

echo "=========================================="
echo "Step 1: Preparing member/non-member sets"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_DIR}/member_data.json" ] && [ -f "${OUTPUT_DIR}/non_member_data.json" ]; then
    echo "⏭️  Data files already exist, skipping..."
    python - <<EOF
import json
with open('${OUTPUT_DIR}/member_data.json', 'r') as f:
    member_data = json.load(f)
with open('${OUTPUT_DIR}/non_member_data.json', 'r') as f:
    non_member_data = json.load(f)
print(f"   Member (SLAKE): {len(member_data)} samples")
print(f"   Non-member (PathVQA): {len(non_member_data)} samples")
EOF
else
    python - <<EOF
import json

# Load MEMBER data (SLAKE training)
print("Loading MEMBER data (SLAKE training)...")
with open('${MEMBER_DATA}', 'r') as f:
    member_data = json.load(f)
print(f"✅ Member data: {len(member_data)} samples")
print(f"   Source: SLAKE training set")

# Load NON-MEMBER data (PathVQA)
print("\nLoading NON-MEMBER data (PathVQA)...")
with open('${NON_MEMBER_DATA}', 'r') as f:
    non_member_data = json.load(f)
print(f"✅ Non-member data: {len(non_member_data)} samples")
print(f"   Source: PathVQA dataset")

# Save for pipeline
member_file = '${OUTPUT_DIR}/member_data.json'
non_member_file = '${OUTPUT_DIR}/non_member_data.json'

with open(member_file, 'w') as f:
    json.dump(member_data, f, indent=2)

with open(non_member_file, 'w') as f:
    json.dump(non_member_data, f, indent=2)

print(f"\n✅ Data prepared:")
print(f"   Member (SLAKE): {len(member_data)} samples")
print(f"   Non-member (PathVQA): {len(non_member_data)} samples")
print(f"   Saved to: ${OUTPUT_DIR}/")
EOF
fi

echo ""

# ============================================================================
# STEP 2: Generate Conversations
# ============================================================================

echo "=========================================="
echo "Step 2: Generating conversations (temperature=0.1)"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_DIR}/member_conversations.json" ]; then
    echo "⏭️  Member conversations already exist, skipping..."
else
    echo "Generating MEMBER conversations (SLAKE)..."
           
    python /scratch1/runhuixu/MIA/slake_mia_conversation_generation.py \
        --model-path "${MODEL_PATH}" \
        --input-json-path "${OUTPUT_DIR}/member_data.json" \
        --image-folder "${MEMBER_IMAGE_FOLDER}" \
        --output-json-path "${OUTPUT_DIR}/member_conversations.json" \
        --conv-mode "${CONV_MODE}" \
        --temperatures 0.1 \
        --repeat 1 \
        --max-new-tokens 512 \
        --use-sampling
fi

echo ""

if [ -f "${OUTPUT_DIR}/non_member_conversations.json" ]; then
    echo "⏭️  Non-member conversations already exist, skipping..."
else
    echo "Generating NON-MEMBER conversations (PathVQA)..."
    python /scratch1/runhuixu/MIA/slake_mia_conversation_generation.py \
        --model-path "${MODEL_PATH}" \
        --input-json-path "${OUTPUT_DIR}/non_member_data.json" \
        --image-folder "${NON_MEMBER_IMAGE_FOLDER}" \
        --output-json-path "${OUTPUT_DIR}/non_member_conversations.json" \
        --conv-mode "${CONV_MODE}" \
        --temperatures 0.1 \
        --repeat 1 \
        --max-new-tokens 512 \
        --use-sampling
fi

echo "✅ Conversations generated"
echo ""

# ============================================================================
# STEP 3: Calculate Similarity Scores (GPU-accelerated)
# ============================================================================

echo "=========================================="
echo "Step 3: Calculating similarity scores"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_DIR}/member_similarity.json" ]; then
    echo "⏭️  Member similarity scores already exist, skipping..."
else
    echo "Calculating similarity for MEMBER set (SLAKE)..."
    python /scratch1/runhuixu//MIA/mia_similarity_calculation.py \
        --conversation-json-path "${OUTPUT_DIR}/member_conversations.json" \
        --similarity-json-path "${OUTPUT_DIR}/member_similarity.json" \
        --temperatures 0.1
fi

echo ""

if [ -f "${OUTPUT_DIR}/non_member_similarity.json" ]; then
    echo "⏭️  Non-member similarity scores already exist, skipping..."
else
    echo "Calculating similarity for NON-MEMBER set (PathVQA)..."
    python /scratch1/runhuixu/MIA/mia_similarity_calculation.py \
        --conversation-json-path "${OUTPUT_DIR}/non_member_conversations.json" \
        --similarity-json-path "${OUTPUT_DIR}/non_member_similarity.json" \
        --temperatures 0.1
fi

echo "✅ Similarity scores calculated"
echo ""

# ============================================================================
# STEP 4: Run Reference Attack
# ============================================================================

echo "=========================================="
echo "Step 4: Running Reference Attack"
echo "=========================================="
echo ""

if [ -f "${OUTPUT_DIR}/attack_results.json" ]; then
    echo "⏭️  Attack results already exist, skipping..."
else
    python /scratch1/runhuixu/MIA/mia_reference_attack.py \
        --member-similarity-file "${OUTPUT_DIR}/member_similarity.json" \
        --non-member-similarity-file "${OUTPUT_DIR}/non_member_similarity.json" \
        --granularity ${GRANULARITY} \
        --temperature 0.1 \
        --similarity-metric ${SIMILARITY_METRIC} \
        --output-file "${OUTPUT_DIR}/attack_results.json"
    echo ""
    echo "✅ Reference attack completed"
fi

echo ""

# ============================================================================
# STEP 5: Display Results
# ============================================================================

echo "=========================================="
echo "REFERENCE ATTACK RESULTS"
echo "=========================================="
echo ""
echo "Member Set: SLAKE training (640 samples)"
echo "Non-Member Set: PathVQA (6,700 samples)"
echo ""

if [ -f "${OUTPUT_DIR}/attack_results.json" ]; then
    python -c "
import json
with open('${OUTPUT_DIR}/attack_results.json', 'r') as f:
    results = json.load(f)

print(f\"AUC:       {results.get('auc', 0):.4f}\")
print(f\"Accuracy:  {results.get('accuracy', 0):.4f}\")
print(f\"Precision: {results.get('precision', 0):.4f}\")
print(f\"Recall:    {results.get('recall', 0):.4f}\")
print(f\"F1 Score:  {results.get('f1', 0):.4f}\")
print(f\"\")
print(f\"Granularity: {results.get('granularity', 50)}\")
print(f\"Members:   {results.get('n_members', 0)}\")
print(f\"Non-members: {results.get('n_non_members', 0)}\")
"
else
    echo "⚠️  Results file not found"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/attack_results.json"
echo ""
echo "Interpreting AUC scores:"
echo "  AUC = 0.5:     No privacy leakage (random guessing)"
echo "  AUC = 0.6-0.7: Weak privacy leakage"
echo "  AUC = 0.7-0.8: Moderate privacy leakage"
echo "  AUC > 0.8:     Strong privacy leakage"
echo ""
echo "Note: Cross-dataset test (SLAKE vs PathVQA)"
echo "      High AUC may indicate domain shift detection"
echo "      rather than pure membership leakage."
echo ""
