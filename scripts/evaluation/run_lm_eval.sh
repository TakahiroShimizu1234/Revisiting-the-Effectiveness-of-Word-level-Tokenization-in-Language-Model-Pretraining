#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-"path/to/model"}
TOKENIZER_PATH=${2:-"$MODEL_PATH"}
OUTPUT_PATH=${3:-"results/lm_eval_results.json"}

TASKS=${TASKS:-"arc_easy,arc_challenge,boolq,piqa,hellaswag,openbookqa"}
DEVICE=${DEVICE:-"cuda:0"}
BATCH_SIZE=${BATCH_SIZE:-"auto"}

python -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},tokenizer=${TOKENIZER_PATH},trust_remote_code=True" \
  --tasks "${TASKS}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --output_path "${OUTPUT_PATH}"
