#!/bin/bash
#PBS -q <gpu_group>
#PBS -l select=1:ngpus=4
#PBS -l walltime=240:00:00
#PBS -W group_list=<account>
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd ${PROJECT_ROOT:-/path/to/project}
source "${VENV_PATH:-venv}/bin/activate"

mkdir -p logs

export HF_HOME="${HF_HOME:-/path/to/huggingface/cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/path/to/huggingface/cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/huggingface/cache/datasets}"

CONFIG="configs/training/scratch/model_1_7b_mp_fwe10k_en_min2_scratch.json"
RUN_DIR="out/pretrain_scratch/mp/1_7b_mp_fwe10k_en_min2_scratch_34b"
TOKENS_TARGET="34000000000"

mkdir -p "${RUN_DIR}"
LOG="logs/pretrain_scratch_1_7b_mp_fwe10k_34b_${PBS_JOBID}.log"

echo "[info] CONFIG=$CONFIG" | tee -a "$LOG"
echo "[info] RUN_DIR=$RUN_DIR" | tee -a "$LOG"
echo "[info] TOKENS_TARGET=$TOKENS_TARGET" | tee -a "$LOG"
date | tee -a "$LOG"
hostname | tee -a "$LOG"

torchrun --nproc_per_node=4 src/pretrain_scratch_unified.py \
  --config "$CONFIG" \
  --output_dir "$RUN_DIR" \
  --resume none \
  --tokens_target "$TOKENS_TARGET" \
  2>&1 | tee -a "$LOG"

echo "[done]" | tee -a "$LOG"
date | tee -a "$LOG"
