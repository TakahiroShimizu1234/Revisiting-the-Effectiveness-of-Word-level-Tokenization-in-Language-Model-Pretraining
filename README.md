# Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining

Code and tokenizer assets for the paper:

**Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining**

Takahiro Shimizu, Tianqi Wang, and Jun Suzuki

**Accepted at the 6th Workshop on Multilingual Representation Learning (MRL 2026), co-located with EMNLP 2026.**

## Overview

This work revisits word-level tokenization for language model pretraining. We compare word-level tokenization with UTF-8 byte fallback against SentencePiece Unigram tokenization across multiple languages, vocabulary sizes, and model configurations.

Both approaches can represent out-of-vocabulary text without relying on an unknown token. We examine downstream accuracy as well as tokenization efficiency and raw-text exposure under fixed emitted-token budgets.

## Experimental Settings

We study four languages:

| Language | Word-level segmentation |
| --- | --- |
| English | Rule-based segmentation |
| French | Language-aware segmentation |
| Chinese | jieba |
| Japanese | MeCab / fugashi |

The primary comparisons use three vocabulary sizes: **10k, 50k, and 100k**.

The English experiments cover six model configurations across the SmolLM2, Gemma 3, Qwen 3, and Llama 3.2 model families. The multilingual experiments extend the comparison to French, Chinese, and Japanese.

## Repository Structure

```text
src/
    Pretraining implementations

tokenizer/
    General word-level tokenizer implementation

tokenizer_assets/
    Word-level and SentencePiece Unigram tokenizer assets
    for English, French, Chinese, and Japanese

scripts/
    Preprocessing, training, evaluation, and tokenization
    analysis scripts

configs/
    Model and training configurations

analysis/
    Additional statistical and token-level analyses

examples/
    Small tokenizer examples

docs/
    Implementation notes
```

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

For downstream evaluation with lm-evaluation-harness, additionally install:

```bash
pip install lm-eval
```

Full pretraining requires substantial computational resources. Some launch scripts were originally written for an HPC environment and must be adapted to other systems.

## Tokenizer Assets

The `tokenizer_assets/` directory contains 24 primary tokenizer conditions:

- Four languages: English, French, Chinese, and Japanese
- Three vocabulary sizes: 10k, 50k, and 100k
- Two tokenizer types: word-level and SentencePiece Unigram

Word-level tokenizer directories contain the tokenizer implementation, vocabulary, and available configuration files.

The SentencePiece Unigram assets use `tokenizer.model` for English, Chinese, and Japanese, and `tokenizer.json` for French.

The word-level tokenizers use language-specific segmentation and UTF-8 byte fallback for out-of-vocabulary units.

### English tokenizer assets

The paper-style English word-level tokenizer assets are:

- `tokenizer_assets/mp_en_paper_10k/`
- `tokenizer_assets/mp_en_paper_50k/`
- `tokenizer_assets/mp_en_paper_100k/`

These assets were copied from the final Llama 3.2 1B English checkpoints for the three nominal vocabulary settings. Their tokenizer files, vocabulary mappings, and sample token-ID sequences were verified against those checkpoints. The saved vocabulary sizes are 10,001, 50,001, and 100,000, respectively.

These assets use rule-based segmentation: alphabetic runs, individual digits, individual punctuation/symbol characters, and one `_` token per whitespace character. Vocabulary matching is lowercase; out-of-vocabulary units fall back to UTF-8 bytes.


Correspondence between these exact vocabulary mappings and every other English model family has not yet been verified.

See `docs/tokenizer.md` for further details.

## Quick Demo

Run the existing small tokenizer example:

```bash
python3 examples/tokenizer_demo.py
```

This demonstrates preservation of in-vocabulary units and byte fallback for out-of-vocabulary units.

## Pretraining

The main pretraining implementations are:

```text
src/pretrain_scratch_unified.py
src/pretrain_scratch_unified_lowercase.py
```

Sanitized example launch scripts and configurations are available under:

```text
scripts/training/pretraining_runs/
configs/training/scratch/
```

The currently included launch scripts and configurations cover the earlier SmolLM2 English runs. They do not constitute a complete set of configurations for every experiment in the camera-ready paper.

The repository does not contain pretrained model checkpoints or raw training corpora.

## Token-Budget Exposure Analysis

The multilingual tokenization analysis script is:

```text
scripts/analysis/analyze_multilingual_tokenization.py
```

It compares word-level and Unigram tokenizers at fixed emitted-token checkpoints, measuring quantities including:

- Documents covered
- Characters and UTF-8 bytes covered
- Segmentation units covered
- Tokens per unit
- Characters and bytes per token
- Byte fallback rate
- Word-level OOV rate

The script uses the tokenizer assets in this repository by default. An alternative project root can be supplied with the `TOKENIZATION_PROJECT_ROOT` environment variable.

For example, a small Chinese smoke test can be run with:

```bash
python3 scripts/analysis/analyze_multilingual_tokenization.py \
  --lang zh \
  --checkpoints 10000 \
  --num_workers 2 \
  --batch_size 32 \
  --fragmentation_docs 20 \
  --max_docs 1000 \
  --output results/zh_smoke_test.csv
```

This is a small workflow check, not a command for reproducing the full-scale results.

## Evaluation

An example zero-shot evaluation command is available at:

```text
scripts/evaluation/run_lm_eval.sh
```

Example:

```bash
bash scripts/evaluation/run_lm_eval.sh \
  path/to/model \
  path/to/tokenizer \
  results/lm_eval_results.json
```

The script provides an example task selection. The exact tasks, preprocessing, tokenizer loading, and scoring settings must be matched to the corresponding experiment before comparing results with the paper.

The English lowercase evaluation patch is provided at:

```text
scripts/evaluation/lowercase_eval_patch/sitecustomize.py
```

## Release Scope

This repository provides tokenizer implementations and assets, the main pretraining implementations, selected launch scripts and configurations, and analysis utilities.

It is not yet a one-command, end-to-end reproduction package for all experiments in the paper. Large model checkpoints, raw corpora, and raw evaluation outputs are not included.

## Citation

If you use this repository, please cite:

**Shimizu, Takahiro; Wang, Tianqi; and Suzuki, Jun. (2026). Revisiting the Effectiveness of Word-level Tokenization in Language Model Pretraining. MRL 2026.**

See `CITATION.cff` for repository citation metadata.

## License

See [LICENSE](LICENSE).
