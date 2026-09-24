# Pretraining Runs

This directory contains sanitized launch scripts for the language model pretraining runs used in the manuscript.

The scripts are provided to document the experimental workflow. They may require adaptation to run on a different cluster environment.

Private server paths, account names, raw corpora, checkpoints, personal email addresses, and cluster-specific identifiers have been removed or replaced with placeholders.

## Contents

- `scratch_pretrain_135m_*`: 135M scratch pretraining runs
- `scratch_pretrain_360m_*`: 360M scratch pretraining runs
- `scratch_pretrain_1_7b_*`: 1.7B scratch pretraining runs

Corresponding sanitized model configuration files are placed under `configs/training/scratch/`.

The main trainer implementations are included under `src/`. Cluster-specific infrastructure, raw corpora, and model checkpoints are not included. The sanitized launch scripts may require adaptation to run in another environment.
