#!/bin/bash
#SBATCH --job-name=pathvqa_3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
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

cd /scratch1/runhuixu/LLaVA-Med/scripts/chunyl/finetune_on_benchmarks

for epoch in 9 ; do
    bash /scratch1/runhuixu/LLaVA-Med/scripts/chunyl/finetune_on_benchmarks/fine_tuning_slake_7B.sh $epoch
    bash /scratch1/runhuixu/LLaVA-Med/scripts/chunyl/finetune_on_benchmarks/eval_slake.sh $epoch

    # outputfile="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-vqarad-finetune/lora/eval/vqa_rad_test_predictions_$epoch.jsonl"
    # outputlog="/project2/ruishanl_1185/SDP_for_VLM/outputs/llava-med-v1.5-vqarad-finetune/lora/eval/accuracy_$epoch.log"
    # python scripts/calculate_acc.py --pred $outputfile  --test /project2/ruishanl_1185/SDP_for_VLM/datasets/vqa_rad/test.json > $outputlog 2>&1
done


