# Pretraining Runs

This directory contains sanitized launch scripts for the language model pretraining runs used in the manuscript.

The scripts are provided to document the experimental workflow. They may require adaptation to run on a different cluster environment.

Private server paths, account names, raw corpora, checkpoints, personal email addresses, and cluster-specific identifiers have been removed or replaced with placeholders.

## Contents

- `scratch_pretrain_135m_*`: 135M scratch pretraining runs
- `scratch_pretrain_360m_*`: 360M scratch pretraining runs
- `scratch_pretrain_1_7b_*`: 1.7B scratch pretraining runs
- `seed_runs/`: additional 135M multi-seed runs
- `docmatch/`: document-matched controls for word-level models
- `docmatch_reverse/`: document-matched controls for subword models

Corresponding sanitized model configuration files are placed under `configs/training/scratch/`.

Note: The launch scripts document the pretraining commands and configurations used in the experiments. The full trainer implementation and cluster-specific infrastructure are not included in this public repository.
