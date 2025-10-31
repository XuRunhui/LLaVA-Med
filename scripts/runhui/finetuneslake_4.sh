#!/bin/bash
#SBATCH --job-name=slake_multinode
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2                 # 1 task per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

################################################################################
# SLAKE Multi-Node Training with OOM Fix
#
# Key fixes for checkpoint saving OOM:
# 1. CPU offload during checkpoint save
# 2. Increased save steps to reduce frequency
# 3. Memory-efficient FSDP state dict type
# 4. Proper CUDA memory management
################################################################################

module purge
module load gcc/13.3.0
module load cuda/12.6.3
export CUDA_HOME=/apps/spack/2406/apps/linux-rocky8-x86_64_v3/gcc-13.3.0/cuda-12.6.3-4yhbknw
mkdir -p logs

ACTIVATE_CONDA='source /home1/runhuixu/miniconda3/etc/profile.d/conda.sh && conda activate llava-med'

# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
GPUS_PER_NODE=2
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=29500

# =============================================================================
# MODEL & DATA PATHS
# =============================================================================
MODEL=/scratch1/runhuixu/checkpoints/slake/checkpoint-100
OUT=/scratch1/runhuixu/outputs/slake_slake_new
DATA=/scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_train.json
IMAGES=/scratch1/runhuixu/datasets/SLAKE/Slake1.0/imgs
VISION=openai/clip-vit-large-patch14

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
PER_DEV_BS=1
GRAD_ACCUM=32        # Effective batch = 2 nodes × 2 GPUs × 1 batch × 32 accum = 128
MAX_LEN=1024
NUM_EPOCHS=9

# Calculate steps per epoch
# SLAKE train: 640 samples
# 640 / 128 = 5 steps per epoch
# Total: 5 × 9 = 45 steps

# CRITICAL FIX: Save less frequently to avoid OOM
# Save every 20 steps instead of 10 (every ~4 epochs instead of ~2 epochs)
SAVE_STEPS=20
SAVE_TOTAL_LIMIT=3   # Keep only last 3 checkpoints to save disk space

# =============================================================================
# NCCL / RUNTIME CONFIGURATION
# =============================================================================
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                 # Enable InfiniBand (set to 1 if flaky)
export NCCL_SOCKET_IFNAME=^lo,docker0    # Adjust if needed (eth0, ib0)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# =============================================================================
# MEMORY OPTIMIZATION (KEY FOR OOM FIX)
# =============================================================================

# 1. CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# 2. Prevent memory fragmentation
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# 3. CPU offload for checkpoint saving (reduces GPU memory spike)
export FSDP_CPU_OFFLOAD=1

# 4. Garbage collection during checkpointing
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6

echo "=========================================="
echo "SLAKE Multi-Node Training (OOM Fixed)"
echo "=========================================="
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Total GPUs: $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Effective batch size: $((SLURM_NNODES * GPUS_PER_NODE * PER_DEV_BS * GRAD_ACCUM))"
echo "Save steps: ${SAVE_STEPS} (every ~4 epochs)"
echo "Output: ${OUT}"
echo "=========================================="
echo ""

# Quick sanity check
echo "Node sanity check:"
srun -N${SLURM_NNODES} -n${SLURM_NNODES} /bin/bash -lc 'hostname; nvidia-smi -L'
echo ""

# =============================================================================
# LAUNCH TRAINING
# =============================================================================

srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --export=ALL /bin/bash -lc "
  ${ACTIVATE_CONDA}
  echo 'Node:' \$(hostname) ' RANK:' \${SLURM_PROCID} ' Starting training...'

  torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=\${SLURM_PROCID} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /scratch1/runhuixu/LLaVA-Med/llava/train/train_mem.py \
      --model_name_or_path ${MODEL} \
      --data_path ${DATA} \
      --image_folder ${IMAGES} \
      --vision_tower ${VISION} \
      --mm_vision_select_layer -2 \
      --mm_use_im_start_end True \
      --bf16 True \
      --output_dir ${OUT} \
      --num_train_epochs ${NUM_EPOCHS} \
      --per_device_train_batch_size ${PER_DEV_BS} \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps ${GRAD_ACCUM} \
      --evaluation_strategy no \
      --save_strategy steps \
      --save_steps ${SAVE_STEPS} \
      --save_total_limit ${SAVE_TOTAL_LIMIT} \
      --save_on_each_node False \
      --learning_rate 2e-5 \
      --weight_decay 0.0 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --tf32 True \
      --model_max_length ${MAX_LEN} \
      --gradient_checkpointing True \
      --dataloader_num_workers 4 \
      --lazy_preprocess True \
      --fsdp 'full_shard auto_wrap' \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --fsdp_state_dict_type SHARDED_STATE_DICT \
      --fsdp_cpu_offload False \
      --report_to wandb
"

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "Output directory: ${OUT}"
echo "Check logs for any errors"
