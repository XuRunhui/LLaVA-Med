#!/bin/bash
#SBATCH --job-name=slake_mia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

################################################################################
# SLAKE-Specific Membership Inference Attack (MIA) Pipeline
#
# This script runs the complete MIA pipeline on your SLAKE fine-tuned model:
# 1. Split data into member/non-member sets
# 2. Generate model responses at different temperatures
# 3. Calculate similarity scores
# 4. Run all three MIA attacks (Reference, Target-Only, Image-Only)
#
# Adapted for YOUR specific LLaVA-Med setup with FSDP checkpoints
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
echo "SLAKE Membership Inference Attack Pipeline"
echo "=========================================="
echo ""

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Model checkpoint (your fine-tuned SLAKE model)
MODEL_PATH="/scratch1/runhuixu/checkpoints/slake/checkpoint-100"

# SLAKE dataset paths
TRAIN_DATA="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_train.json"  # Member set (used in training)
TEST_DATA="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_test.json"    # Non-member set (held out)
IMAGE_FOLDER="/scratch1/runhuixu/datasets/SLAKE/Slake1.0/imgs"

# Output directory
OUTPUT_DIR="/scratch1/runhuixu/mia_results/slake_checkpoint100"
mkdir -p "$OUTPUT_DIR"

# Attack parameters
GRANULARITY=50  # Number of samples per statistical test
SIMILARITY_METRIC="rouge2_f"  # Options: rouge1_f, rouge2_f, rougeL_f, bleu, bertscore_f

# Conversation mode (use "simple" for your model)
CONV_MODE="simple"

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Member data (training): ${TRAIN_DATA}"
echo "  Non-member data (test): ${TEST_DATA}"
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

python - <<EOF
import json

# Load training data (MEMBER SET - model was trained on this)
print("Loading training data (member set)...")
with open('${TRAIN_DATA}', 'r') as f:
    member_data = json.load(f)
print(f"✅ Member data: {len(member_data)} samples (used in training)")

# Load test data (NON-MEMBER SET - model never saw this)
print("Loading test data (non-member set)...")
with open('${TEST_DATA}', 'r') as f:
    non_member_data = json.load(f)
print(f"✅ Non-member data: {len(non_member_data)} samples (held out)")

# Save for pipeline
member_file = '${OUTPUT_DIR}/member_data.json'
non_member_file = '${OUTPUT_DIR}/non_member_data.json'

with open(member_file, 'w') as f:
    json.dump(member_data, f, indent=2)

with open(non_member_file, 'w') as f:
    json.dump(non_member_data, f, indent=2)

print(f"\n✅ Data prepared:")
print(f"   Member (train): {len(member_data)} samples")
print(f"   Non-member (test): {len(non_member_data)} samples")
print(f"   Total: {len(member_data) + len(non_member_data)} samples")
print(f"   Saved to: ${OUTPUT_DIR}/")
EOF

echo ""

# ============================================================================
# STEP 2: Generate Conversations for Reference Attack
# ============================================================================

echo "=========================================="
echo "Step 2: Reference Attack - Generating conversations"
echo "=========================================="
echo ""

echo "Generating member conversations (temperature=0.1)..."
python /scratch1/runhuixu/MIA/slake_mia_conversation_generation.py \
    --model-path "${MODEL_PATH}" \
    --input-json-path "${OUTPUT_DIR}/member_data.json" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-json-path "${OUTPUT_DIR}/reference/member_conversations.json" \
    --conv-mode "${CONV_MODE}" \
    --temperatures 0.1 \
    --repeat 1 \
    --max-new-tokens 512 \
    --use-sampling

echo ""
echo "Generating non-member conversations (temperature=0.1)..."
python /scratch1/runhuixu/MIA/slake_mia_conversation_generation.py \
    --model-path "${MODEL_PATH}" \
    --input-json-path "${OUTPUT_DIR}/non_member_data.json" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-json-path "${OUTPUT_DIR}/reference/non_member_conversations.json" \
    --conv-mode "${CONV_MODE}" \
    --temperatures 0.1 \
    --repeat 1 \
    --max-new-tokens 512 \
    --use-sampling

echo "✅ Reference attack conversations generated"
echo ""

# # ============================================================================
# # STEP 3: Generate Conversations for Target-Only Attack
# # ============================================================================

# echo "=========================================="
# echo "Step 3: Target-Only Attack - Generating conversations"
# echo "=========================================="
# echo ""

# echo "Generating member conversations (temperatures=0.1, 1.5)..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/slake_mia_conversation_generation.py \
#     --model-path "${MODEL_PATH}" \
#     --input-json-path "${OUTPUT_DIR}/member_data.json" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --output-json-path "${OUTPUT_DIR}/target_only/member_conversations.json" \
#     --conv-mode "${CONV_MODE}" \
#     --temperatures 0.1 1.5 \
#     --repeat 1 \
#     --max-new-tokens 512 \
#     --use-sampling

# echo ""
# echo "Generating non-member conversations (temperatures=0.1, 1.5)..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/slake_mia_conversation_generation.py \
#     --model-path "${MODEL_PATH}" \
#     --input-json-path "${OUTPUT_DIR}/non_member_data.json" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --output-json-path "${OUTPUT_DIR}/target_only/non_member_conversations.json" \
#     --conv-mode "${CONV_MODE}" \
#     --temperatures 0.1 1.5 \
#     --repeat 1 \
#     --max-new-tokens 512 \
#     --use-sampling

# echo "✅ Target-only attack conversations generated"
# echo ""

# ============================================================================
# STEP 4: Generate Conversations for Image-Only Attack
# ============================================================================

# echo "=========================================="
# echo "Step 4: Image-Only Attack - Generating conversations"
# echo "=========================================="
# echo ""

# echo "Generating member conversations (temperature=0.1, 5 repetitions)..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/slake_mia_conversation_generation.py \
#     --model-path "${MODEL_PATH}" \
#     --input-json-path "${OUTPUT_DIR}/member_data.json" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --output-json-path "${OUTPUT_DIR}/image_only/member_conversations.json" \
#     --conv-mode "${CONV_MODE}" \
#     --temperatures 0.1 \
#     --repeat 5 \
#     --max-new-tokens 512 \
#     --use-sampling

# echo ""
# echo "Generating non-member conversations (temperature=0.1, 5 repetitions)..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/slake_mia_conversation_generation.py \
#     --model-path "${MODEL_PATH}" \
#     --input-json-path "${OUTPUT_DIR}/non_member_data.json" \
#     --image-folder "${IMAGE_FOLDER}" \
#     --output-json-path "${OUTPUT_DIR}/image_only/non_member_conversations.json" \
#     --conv-mode "${CONV_MODE}" \
#     --temperatures 0.1 \
#     --repeat 5 \
#     --max-new-tokens 512 \
#     --use-sampling

# echo "✅ Image-only attack conversations generated"
# echo ""

# ============================================================================
# STEP 5: Calculate Similarity Scores
# ============================================================================

echo "=========================================="
echo "Step 5: Calculating similarity scores"
echo "=========================================="
echo ""

# Reference Attack similarity
echo "Calculating similarity for Reference Attack..."
python /scratch1/runhuixu/MIA/mia_similarity_calculation.py \
    --conversation-json-path "${OUTPUT_DIR}/reference/member_conversations.json" \
    --similarity-json-path "${OUTPUT_DIR}/reference/member_similarity.json" \
    --temperatures 0.1

python /scratch1/runhuixu/MIA/mia_similarity_calculation.py \
    --conversation-json-path "${OUTPUT_DIR}/reference/non_member_conversations.json" \
    --similarity-json-path "${OUTPUT_DIR}/reference/non_member_similarity.json" \
    --temperatures 0.1

# Target-Only Attack similarity
# echo "Calculating similarity for Target-Only Attack..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_similarity_calculation.py \
#     --conversation-json-path "${OUTPUT_DIR}/target_only/member_conversations.json" \
#     --similarity-json-path "${OUTPUT_DIR}/target_only/member_similarity.json" \
#     --temperatures 0.1 1.5

# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_similarity_calculation.py \
#     --conversation-json-path "${OUTPUT_DIR}/target_only/non_member_conversations.json" \
#     --similarity-json-path "${OUTPUT_DIR}/target_only/non_member_similarity.json" \
#     --temperatures 0.1 1.5

# # Image-Only Attack similarity (pairwise)
# echo "Calculating pairwise similarity for Image-Only Attack..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_similarity_repeating.py \
#     --conversation-json-path "${OUTPUT_DIR}/image_only/member_conversations.json" \
#     --similarity-json-path "${OUTPUT_DIR}/image_only/member_similarity.json" \
#     --temperatures 0.1 \
#     --repeating-num 5

# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_similarity_repeating.py \
#     --conversation-json-path "${OUTPUT_DIR}/image_only/non_member_conversations.json" \
#     --similarity-json-path "${OUTPUT_DIR}/image_only/non_member_similarity.json" \
#     --temperatures 0.1 \
#     --repeating-num 5

echo "✅ Similarity scores calculated"
echo ""

# ============================================================================
# STEP 6: Run MIA Attacks
# ============================================================================

echo "=========================================="
echo "Step 6: Running MIA attacks"
echo "=========================================="
echo ""

# Reference Attack
echo "Running Reference Attack..."
python /scratch1/runhuixu/MIA/mia_reference_attack.py \
    --member-similarity-file "${OUTPUT_DIR}/reference/member_similarity.json" \
    --non-member-similarity-file "${OUTPUT_DIR}/reference/non_member_similarity.json" \
    --granularity ${GRANULARITY} \
    --temperature 0.1 \
    --similarity-metric ${SIMILARITY_METRIC} \
    --output-file "${OUTPUT_DIR}/reference/attack_results.json"

python /scratch1/runhuixu/MIA/mia_reference_attack.py \
    --member-similarity-file "/scratch1/runhuixu/mia_results/slake_checkpoint110/reference/member_similarity.json" \
    --non-member-similarity-file "/scratch1/runhuixu/mia_results/slake_checkpoint110/reference/non_member_similarity.json" \
    --granularity 50 \
    --temperature 0.1 \
    --similarity-metric rouge2_f \
    --output-file "/scratch1/runhuixu/mia_results/slake_checkpoint110/reference/attack_results.json"
# Target-Only Attack
# echo "Running Target-Only Attack..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_target_only_attack.py \
#     --member-similarity-file "${OUTPUT_DIR}/target_only/member_similarity.json" \
#     --non-member-similarity-file "${OUTPUT_DIR}/target_only/non_member_similarity.json" \
#     --granularity ${GRANULARITY} \
#     --temperature-low 0.1 \
#     --temperature-high 1.5 \
#     --similarity-metric ${SIMILARITY_METRIC} \
#     --output-file "${OUTPUT_DIR}/target_only/attack_results.json"

# # Image-Only Attack
# echo "Running Image-Only Attack..."
# python /scratch1/runhuixu/LLaVA-Med/MIA/mia_image_only_attack.py \
#     --member-similarity-file "${OUTPUT_DIR}/image_only/member_similarity.json" \
#     --non-member-similarity-file "${OUTPUT_DIR}/image_only/non_member_similarity.json" \
#     --granularity ${GRANULARITY} \
#     --temperature 0.1 \
#     --similarity-metric ${SIMILARITY_METRIC} \
#     --output-file "${OUTPUT_DIR}/image_only/attack_results.json"

echo "✅ All attacks completed"
echo ""

# ============================================================================
# STEP 7: Display Results
# ============================================================================

echo "=========================================="
echo "ATTACK RESULTS SUMMARY"
echo "=========================================="
echo ""
#  target_only image_only
for attack_type in reference ; do
    result_file="${OUTPUT_DIR}/${attack_type}/attack_results.json"

    if [ -f "$result_file" ]; then
        echo "--- ${attack_type^^} ATTACK ---"
        python -c "
import json
with open('${result_file}', 'r') as f:
    results = json.load(f)

print(f\"  AUC:       {results.get('auc', 0):.4f} ± {results.get('auc_std', 0):.4f}\")
print(f\"  Accuracy:  {results.get('accuracy', 0):.4f} ± {results.get('accuracy_std', 0):.4f}\")
print(f\"  Precision: {results.get('precision', 0):.4f}\")
print(f\"  Recall:    {results.get('recall', 0):.4f}\")
print(f\"  F1 Score:  {results.get('f1', 0):.4f}\")
"
        echo ""
    else
        echo "--- ${attack_type^^} ATTACK ---"
        echo "  ⚠️  Results file not found"
        echo ""
    fi
done

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results directory: ${OUTPUT_DIR}"
echo ""
echo "Files generated:"
echo "  ${OUTPUT_DIR}/reference/attack_results.json"
echo "  ${OUTPUT_DIR}/target_only/attack_results.json"
echo "  ${OUTPUT_DIR}/image_only/attack_results.json"
echo ""
echo "Interpreting AUC scores:"
echo "  AUC = 0.5:     No privacy leakage (random guessing)"
echo "  AUC = 0.6-0.7: Weak privacy leakage"
echo "  AUC = 0.7-0.8: Moderate privacy leakage"
echo "  AUC > 0.8:     Strong privacy leakage"
echo ""
