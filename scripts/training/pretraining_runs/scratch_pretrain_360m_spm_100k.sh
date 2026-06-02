#!/bin/bash
#PBS -N 360m_spm100k
#PBS -q <gpu_group>
#PBS -l select=1:ngpus=4
#PBS -l walltime=100:00:00
#PBS -W group_list=<account>
#PBS -j oe
#PBS -m abe

cd ${PROJECT_ROOT:-/path/to/project} || exit 1
source "${VENV_PATH:-venv}/bin/activate"

python src/pretrain_scratch_unified.py \
  --config configs/training/scratch/model_360m_spm_100k_en_scratch.json \
  --output_dir out/pretrain_scratch/spm/360m_spm_100k_en_scratch_7p2b \
  --tokens_target 7200000000 \
  --save_steps 50000 \
  --logging_steps 50 \
  --resume none
