#!/bin/bash
#PBS -N 135m_mp10k
#PBS -q <gpu_group>
#PBS -l select=1:ngpus=4
#PBS -l walltime=100:00:00
#PBS -W group_list=<account>
#PBS -j oe
#PBS -m abe

cd ${PROJECT_ROOT:-/path/to/project} || exit 1
source "${VENV_PATH:-venv}/bin/activate"

python src/pretrain_scratch_unified.py \
  --config configs/training/scratch/model_135m_mp_fwe10k_en_min2_scratch.json \
  --output_dir out/pretrain_scratch/mp/135m_mp_fwe10k_en_min2_scratch_2p7b \
  --tokens_target 2700000000 \
  --save_steps 50000 \
  --logging_steps 50 \
  --resume none
