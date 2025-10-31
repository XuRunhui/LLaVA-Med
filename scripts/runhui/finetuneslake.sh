#!/bin/bash
#SBATCH --job-name=slake_2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw

mkdir -p logs

# Conda (batch-safe) activation
source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh
conda activate llava-med

# Configuration
GPUS=2  # Number of GPUs you have (adjust this)
model_name_or_path=/scratch1/runhuixu/LLaVA-7b-v0 # ← CRITICAL: Use Stage 2 

output_dir=/scratch1/runhuixu/outputs/slake_fulltuning
data_path=/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_train.json
image_folder=/scratch1/runhuixu/datasets/SLAKE/Slake1.0/imgs
vision_tower=openai/clip-vit-large-patch14

# Calculate gradient accumulation to achieve global batch size of 128
# Global batch size = GPUS × per_device_train_batch_size × gradient_accumulation_steps

# 128 = 4 × 2 × 16  (adjust based on your GPU memory)
per_device_batch_size=1  # Start with 2, reduce to 1 if OOM
gradient_accumulation=64  # Adjust to maintain global batch size = 128

torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=25001 \
    /scratch1/runhuixu/LLaVA-Med/llava/train/train_mem.py \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --vision_tower ${vision_tower} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${gradient_accumulation} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb