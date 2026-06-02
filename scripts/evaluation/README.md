# Evaluation Scripts

This directory contains example commands for downstream evaluation.

The reported experiments used zero-shot evaluation with `lm-evaluation-harness`.
Large model checkpoints and raw evaluation outputs are not included in this repository.

## Example

```bash
bash scripts/evaluation/run_lm_eval.sh path/to/model path/to/tokenizer results/lm_eval_results.json
```

By default, the script evaluates:

```text
arc_easy, arc_challenge, boolq, piqa, hellaswag, openbookqa
```

You can override tasks with:

```bash
TASKS="arc_easy,arc_challenge" bash scripts/evaluation/run_lm_eval.sh path/to/model
```
