


python /scratch1/runhuixu/LLaVA-Med/scripts/chunyl/finetune_on_benchmarks/evaluate_metrics.py \
  --pred /scratch1/runhuixu/evaluation/test/slake/test-answer-file.jsonl \
  --test  /scratch1/runhuixu/datasets/SLAKE/Slake1.0/llava_test.jsonl \
  --report metrics.json


# python /scratch1/runhuixu/LLaVA-Med/scripts/chunyl/finetune_on_benchmarks/evaluate_metrics.py \
#   --pred /scratch1/runhuixu/outputs/pathvqa/eval/test_result_9.jsonl \
#   --test  /scratch1/runhuixu/datasets/PathVQA/annotations/llava_test.jsonl \
#   --report metrics.json