# src/pretrain_scratch_unified.py
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import Dict, Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)

# -----------------------
# Config (user json + extras)
# -----------------------

@dataclass
class RunConfig:
    # from your configs/model_*.json
    model_path: str
    tokenizer_path: str
    context_length: int
    batch_size: int  # treat as GLOBAL batch size
    learning_rate: float

    # defaults for pretrain
    dataset: str = "HuggingFaceTB/smollm-corpus"
    subset: str = "fineweb-edu-dedup"
    split: str = "train"
    text_key: str = "text"
    shuffle_buffer: int = 50_000
    seed: int = 42

    # training control
    tokens_target: int = 5_000_000  # per job
    target_docs: Optional[int] = None  # stop after reading this many source documents
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    bf16: bool = True

    # io/log
    output_dir: str = "out/pretrain_scratch/run_000"
    save_steps: int = 200
    logging_steps: int = 10
    save_total_limit: int = 2

    # resume
    resume: str = "auto"  # auto | none | /path/to/checkpoint-xxxx

    # tokenizer safety
    tokenizer_chunk_max: int = 8192


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        base = json.load(f)
    if isinstance(base.get("learning_rate"), str):
        base["learning_rate"] = float(base["learning_rate"])
    return RunConfig(**base)


def world_size() -> int:
    return max(int(os.environ.get("WORLD_SIZE", "1")), 1)


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    if not output_dir.exists():
        return None
    ckpts = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not ckpts:
        return None

    def step_of(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=step_of)
    return str(ckpts[-1])


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def safe_tokenizer_chunk_max(tokenizer, fallback: int = 8192) -> int:
    m = getattr(tokenizer, "model_max_length", None)
    if m is None:
        return fallback
    try:
        m_int = int(m)
    except Exception:
        return fallback
    if m_int <= 0 or m_int > 100_000:
        return fallback
    return m_int


def load_local_mp_tokenizer(tok_dir: str):
    """
    Load MPTokenizer from a local tokenizer directory:
      tok_dir/
        mp_tokenizer.py
        vocab.json
    """
    tok_dir = str(tok_dir)
    mp_path = Path(tok_dir) / "mp_tokenizer.py"
    vocab_path = Path(tok_dir) / "vocab.json"
    if not mp_path.exists() or not vocab_path.exists():
        raise FileNotFoundError(f"mp_tokenizer.py or vocab.json not found in {tok_dir}")

    spec = importlib.util.spec_from_file_location("mp_tokenizer_local", str(mp_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    MPTokenizer = getattr(module, "MPTokenizer")
    return MPTokenizer(vocab_file=str(vocab_path))


def load_tokenizer_unified(tokenizer_path: str):
    """
    Unified tokenizer loader:
      - If tokenizer_path is a dir with mp_tokenizer.py -> load MPTokenizer (slow/custom)
      - Else -> AutoTokenizer.from_pretrained (SPM etc.)
    """
    p = Path(tokenizer_path)
    if p.is_dir() and (p / "mp_tokenizer.py").exists():
        tok = load_local_mp_tokenizer(str(p))
        kind = "mp"
        # MP is slow; doesn't support HF "fast" features.
        # Ensure pad token exists for Trainer.
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok, kind
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=(False if __import__('os').path.exists(__import__('os').path.join(tokenizer_path,'tokenizer.model')) else True))
        kind = "spm"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok, kind


# -----------------------
# Streaming -> tokenize -> pack
# -----------------------

class PackedStreamingDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset -> tokenize -> concatenate -> pack into fixed length blocks.
    Yields {"input_ids","labels","attention_mask"} with shape [context_length].
    """

    def __init__(
        self,
        dataset_name: str,
        subset: str,
        split: str,
        text_key: str,
        tokenizer,
        context_length: int,
        shuffle_buffer: int,
        seed: int,
        add_eos_between_docs: bool = True,
        tokenizer_chunk_max: int = 8192,
        target_docs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.text_key = text_key
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.add_eos_between_docs = add_eos_between_docs
        self.tokenizer_chunk_max = tokenizer_chunk_max
        self.target_docs = target_docs

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ds = load_dataset(self.dataset_name, self.subset, split=self.split, streaming=True)
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        eos_id = self.tokenizer.eos_token_id
        chunk_max = safe_tokenizer_chunk_max(self.tokenizer, fallback=self.tokenizer_chunk_max)

        buf: List[int] = []
        docs_seen = 0
        for ex in ds:
            text = ex.get(self.text_key, None)
            if not text:
                continue

            # English controlled condition:
            # lowercase before tokenization during language-model pretraining.
            text = text.lower()

            docs_seen += 1

            chunks = None
            try:
                enc = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=chunk_max,
                    return_overflowing_tokens=True,
                    return_attention_mask=False,
                )
                chunks = enc.get("input_ids", None)
                if chunks and isinstance(chunks, list) and chunks and isinstance(chunks[0], int):
                    chunks = [chunks]
            except Exception:
                chunks = None

            if not chunks:
                chunks = []
                char_chunk = 4096
                for i in range(0, len(text), char_chunk):
                    part = text[i:i + char_chunk]
                    enc2 = self.tokenizer(
                        part,
                        add_special_tokens=False,
                        return_attention_mask=False,
                    )
                    ids = enc2.get("input_ids", None)
                    if not ids:
                        continue
                    if isinstance(ids, list) and ids and isinstance(ids[0], list):
                        chunks.extend(ids)
                    else:
                        chunks.append(ids)

            if not chunks:
                continue

            for ids in chunks:
                if not ids:
                    continue

                if self.add_eos_between_docs and eos_id is not None:
                    ids = ids + [eos_id]

                buf.extend(ids)

                while len(buf) >= self.context_length:
                    block = buf[: self.context_length]
                    buf = buf[self.context_length :]

                    input_ids = torch.tensor(block, dtype=torch.long)
                    labels = input_ids.clone()
                    attention_mask = torch.ones_like(input_ids)

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                        "docs_seen": torch.tensor(docs_seen, dtype=torch.long),
                    }


# -----------------------
# Meta logger
# -----------------------

class MetaLogger(TrainerCallback):
    def __init__(self, meta_path: Path, tokens_per_step: int, tokens_target: int) -> None:
        self.meta_path = meta_path
        self.tokens_per_step = tokens_per_step
        self.tokens_target = tokens_target
        self.t0 = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        if is_rank0():
            write_json(
                self.meta_path,
                {
                    "status": "running",
                    "start_time_unix": self.t0,
                    "tokens_target": self.tokens_target,
                    "tokens_per_step": self.tokens_per_step,
                    "max_steps": int(args.max_steps),
                },
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not is_rank0():
            return
        t1 = time.time()
        steps = int(state.global_step)
        consumed = steps * self.tokens_per_step
        elapsed = t1 - self.t0
        tps = consumed / elapsed if elapsed > 0 else None
        write_json(
            self.meta_path,
            {
                "status": "finished",
                "start_time_unix": self.t0,
                "end_time_unix": t1,
                "elapsed_sec": elapsed,
                "tokens_target": self.tokens_target,
                "tokens_per_step": self.tokens_per_step,
                "global_step": steps,
                "consumed_tokens_est": consumed,
                "tokens_per_sec_est": tps,
                "output_dir": str(Path(args.output_dir).resolve()),
            },
        )


# -----------------------
# Doc-matched Trainer
# -----------------------

class DocMatchTrainer(Trainer):
    def __init__(self, *args, target_docs: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_docs = target_docs

    def training_step(self, model, inputs, *args, **kwargs):
        docs_seen = inputs.pop("docs_seen", None)
        loss = super().training_step(model, inputs, *args, **kwargs)

        if self.target_docs is not None and docs_seen is not None:
            try:
                current_docs = int(docs_seen.max().item())
            except Exception:
                current_docs = int(docs_seen)
            if current_docs >= self.target_docs:
                if is_rank0():
                    print(f"[docmatch] reached target_docs: {current_docs} >= {self.target_docs}")
                self.control.should_training_stop = True

        return loss


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="configs/model_*.json")
    ap.add_argument("--output_dir", type=str, default=None, help="override output_dir")
    ap.add_argument("--tokens_target", type=int, default=None, help="override tokens_target per job")
    ap.add_argument("--target_docs", type=int, default=None, help="stop after reading this many source documents")
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--logging_steps", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="auto|none|/path/to/checkpoint-*")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_run_config(args.config)

    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.tokens_target is not None:
        cfg.tokens_target = args.tokens_target
    if args.target_docs is not None:
        cfg.target_docs = args.target_docs
    if args.save_steps is not None:
        cfg.save_steps = args.save_steps
    if args.logging_steps is not None:
        cfg.logging_steps = args.logging_steps
    if args.resume is not None:
        cfg.resume = args.resume
    if args.seed is not None:
        cfg.seed = args.seed

    set_seed(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer (MP or SPM)
    tokenizer, tok_kind = load_tokenizer_unified(cfg.tokenizer_path)

    # model (SCRATCH)
    # load ONLY config from model_path, then random init
    config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=True)
    config.vocab_size = len(tokenizer)

    # Fairness/safety: vocab_size must match
    model_vocab = getattr(config, "vocab_size", None)
    tok_vocab = len(tokenizer)
    if model_vocab is not None and model_vocab != tok_vocab:
        raise ValueError(
            f"[scratch] vocab_size mismatch: config.vocab_size={model_vocab} vs len(tokenizer)={tok_vocab}. "
            "Make them equal (e.g., both 49152) for scratch comparison."
        )

    # align special ids (no vocab change)
    if tokenizer.bos_token_id is not None:
        config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None:
        config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    # quick id-range sanity check (catches "_" missing bug etc.)
    probe_ids = tokenizer("hello world", add_special_tokens=False).get("input_ids", [])
    if probe_ids:
        mx = max(probe_ids)
        if mx >= tok_vocab:
            raise RuntimeError(f"[scratch] tokenizer produced id={mx} but vocab_size={tok_vocab} (broken tokenizer)")

    # dataset (packed streaming)
    train_ds = PackedStreamingDataset(
        dataset_name=cfg.dataset,
        subset=cfg.subset,
        split=cfg.split,
        text_key=cfg.text_key,
        tokenizer=tokenizer,
        context_length=cfg.context_length,
        shuffle_buffer=cfg.shuffle_buffer,
        seed=cfg.seed,
        add_eos_between_docs=True,
        tokenizer_chunk_max=cfg.tokenizer_chunk_max,
        target_docs=cfg.target_docs,
    )

    ws = world_size()

    # Interpret cfg.batch_size as GLOBAL batch size (across all GPUs)
    per_device = max(1, cfg.batch_size // ws)
    grad_accum = max(1, math.ceil(cfg.batch_size / (per_device * ws)))
    global_batch = per_device * grad_accum * ws

    tokens_per_step = global_batch * cfg.context_length
    max_steps = math.ceil(cfg.tokens_target / tokens_per_step)

    # resume
    resume_from: Optional[str] = None
    if cfg.resume == "none":
        resume_from = None
    elif cfg.resume == "auto":
        resume_from = find_latest_checkpoint(out_dir)
    else:
        resume_from = cfg.resume

    if is_rank0():
        write_json(out_dir / "config_effective.json", cfg.__dict__)
        print("[cfg] mode: SCRATCH (unified)")
        print("[cfg] tok_kind:", tok_kind)
        print("[cfg] model_path:", cfg.model_path)
        print("[cfg] tokenizer_path:", cfg.tokenizer_path)
        print("[cfg] vocab_size:", tok_vocab)
        print("[cfg] world_size:", ws)
        print("[cfg] per_device_batch_size:", per_device)
        print("[cfg] grad_accum:", grad_accum)
        print("[cfg] global_batch:", global_batch, "(requested:", cfg.batch_size, ")")
        print("[cfg] context_length:", cfg.context_length)
        print("[cfg] tokens_per_step:", tokens_per_step)
        print("[cfg] tokens_target:", cfg.tokens_target)
        print("[cfg] target_docs:", cfg.target_docs)
        print("[cfg] max_steps:", max_steps)
        print("[resume] from:", resume_from)
        print("[tok] tokenizer_chunk_max:", safe_tokenizer_chunk_max(tokenizer, cfg.tokenizer_chunk_max))

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        bf16=cfg.bf16,
        fp16=False,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        dataloader_num_workers=2,
        report_to=["wandb"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    meta_path = out_dir / "train_meta.json"

    trainer_cls = DocMatchTrainer if cfg.target_docs is not None else Trainer

    trainer = trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        callbacks=[MetaLogger(meta_path, tokens_per_step=tokens_per_step, tokens_target=cfg.tokens_target)],
        target_docs=cfg.target_docs,
    ) if cfg.target_docs is not None else trainer_cls(
        model=model,
        args=targs,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        callbacks=[MetaLogger(meta_path, tokens_per_step=tokens_per_step, tokens_target=cfg.tokens_target)],
    )

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(out_dir / "final"))
    if is_rank0():
        tokenizer.save_pretrained(str(out_dir / "final"))
        print("Saved final to:", out_dir / "final")


if __name__ == "__main__":
    main()