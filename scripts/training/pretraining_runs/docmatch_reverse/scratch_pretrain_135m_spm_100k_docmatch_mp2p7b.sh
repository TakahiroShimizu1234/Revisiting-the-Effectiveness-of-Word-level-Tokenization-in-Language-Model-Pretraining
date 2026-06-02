#!/bin/bash
#PBS -N 135m_spm100k_docmatch_mp
#PBS -q sg
#PBS -l select=1:ngpus=4
#PBS -l walltime=100:00:00
#PBS -W group_list=<account>
#PBS -j oe
#PBS -m abe

cd ${PROJECT_ROOT:-/path/to/project} || exit 1
source "${VENV_PATH:-venv}/bin/activate"

python src/pretrain_scratch_unified.py \
  --config configs/training/scratch/model_135m_spm_100k_en_scratch.json \
  --output_dir out/pretrain_scratch/spm/135m_spm_100k_en_docmatch_mp2p7b \
  --target_docs 2650000 \
  --save_steps 50000 \
  --logging_steps 50
