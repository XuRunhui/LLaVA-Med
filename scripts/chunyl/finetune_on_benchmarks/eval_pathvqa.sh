
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <EPOCH_NUMBER>"
    exit 1
fi

NUM_EPOCHS=$1
EVALUATION_FILE="/scratch1/runhuixu/outputs/pathvqa/eval/test_result_$NUM_EPOCHS.jsonl"

python /scratch1/runhuixu/LLaVA-Med/llava/eval/model_vqa_med.py  --model-name /scratch1/runhuixu/outputs/pathvqa \
    --question-file \
    /scratch1/runhuixu/datasets/PathVQA/annotations/llava_test.json\
    --image-folder /scratch1/runhuixu/datasets/PathVQA/images \
    --answers-file $EVALUATION_FILE
