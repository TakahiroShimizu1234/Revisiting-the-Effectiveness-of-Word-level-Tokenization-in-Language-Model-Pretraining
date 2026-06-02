# Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining

This repository contains the main code used for the experiments in:

**Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining**

## Overview

This project revisits word-level tokenization for language model pretraining.
We compare word-level tokenization with byte fallback against subword tokenization methods such as SentencePiece unigram and BPE.

## Repository Structure

```text
tokenizer/   Word-level tokenizer implementation with byte fallback
analysis/    OOV analysis and plotting scripts
scripts/     Utility scripts for experiment workflows
configs/     Configuration files
examples/    Small examples for tokenizer behavior
results/     Processed result files used for analysis
docs/        Additional notes and documentation
```

## Main Components

- Word-level tokenizer with byte fallback
- OOV-count analysis scripts
- Plotting scripts for OOV count and downstream accuracy
- Skeleton directories for experiment scripts, configs, examples, and processed results

## Reproducing Experiments

Full language model pretraining may require large-scale GPU resources.
This repository provides the main tokenizer implementation and analysis scripts used for the reported experiments.

## Citation

This repository accompanies the paper **Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining**.

Citation information will be added after the paper is finalized.

## Quick Demo

Run the tokenizer demo with:

```bash
python3 examples/tokenizer_demo.py
```

The demo shows how the tokenizer preserves known word-level units and applies UTF-8 byte fallback to out-of-vocabulary units.

Example output:

```text
Input : cats sit on sofa
Tokens: ['cats', '_', 'sit', '_', 'on', '_', '<0x73>', '<0x6F>', '<0x66>', '<0x61>']
Decode: cats sit on sofa
```

## Code Used in the Paper

This repository includes scripts used for the analyses reported in the paper:

- `analysis/statistics/`: Wilcoxon signed-rank tests for multi-seed results
- `analysis/logprob/`: token-level log-probability analysis and visualization
- `scripts/checks/`: vocabulary sanity checks
- `configs/model/`: model configuration files used in the experiments

Large pretrained checkpoints, raw training corpora, and external libraries are not included.

## Preprocessing and Evaluation Scripts

This repository also includes lightweight scripts for reproducing key parts of the experimental workflow:

- `scripts/preprocessing/build_vocab.py`: builds a word-level vocabulary with reserved special tokens and UTF-8 byte fallback tokens
- `scripts/preprocessing/train_sentencepiece.py`: trains SentencePiece unigram or BPE tokenizer baselines
- `scripts/evaluation/run_lm_eval.sh`: provides an example `lm-evaluation-harness` command for downstream zero-shot evaluation

Example vocabulary construction:

```bash
python3 scripts/preprocessing/build_vocab.py --input examples/sample_text.txt --output examples/demo_vocab.json --vocab_size 300
```

Example SentencePiece baseline training:

```bash
python3 scripts/preprocessing/train_sentencepiece.py --input examples/sample_text.txt --model_prefix examples/demo_spm_unigram --vocab_size 300 --model_type unigram --byte_fallback
```

Example downstream evaluation:

```bash
bash scripts/evaluation/run_lm_eval.sh path/to/model path/to/tokenizer results/lm_eval_results.json
```
