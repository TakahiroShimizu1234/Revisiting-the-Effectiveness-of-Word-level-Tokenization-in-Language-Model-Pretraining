#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import multiprocessing as mp
import json
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from datasets import load_dataset


ROOT = Path(os.environ.get("TOKENIZATION_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

VOCABS = ("10k", "50k", "100k")
BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
WS_RE = re.compile(r"\s+|\S+", re.UNICODE)

_LANG = None
_WORD_VOCABS = {}
_SPM = {}
_EN_TOKENIZER = None
_FR_MODULE = None
_JA_TAGGER = None


# ============================================================
# SentencePiece / tokenizer.json backend
# ============================================================

class SubwordBackend:
    def __init__(self, path):
        path = str(path)
        self.path = path

        if path.endswith(".model"):
            import sentencepiece as spm

            self.kind = "sentencepiece"
            self.obj = spm.SentencePieceProcessor(model_file=path)

        elif path.endswith(".json"):
            from tokenizers import Tokenizer

            self.kind = "tokenizers"
            self.obj = Tokenizer.from_file(path)

        else:
            raise ValueError(f"Unsupported tokenizer file: {path}")

    def pieces(self, text):
        if self.kind == "sentencepiece":
            return self.obj.encode(text, out_type=str)

        enc = self.obj.encode(text, add_special_tokens=False)
        return enc.tokens


# ============================================================
# Dynamic loader
# ============================================================

def load_python_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Paths
# ============================================================

def word_dir(lang, vocab):
    if lang == "en":
        return ROOT / f"tokenizer_assets/mp_en_whitespace_{vocab}"
    if lang == "fr":
        return ROOT / f"tokenizer_assets/mp_fr_langaware_{vocab}"
    if lang == "zh":
        return ROOT / f"tokenizer_assets/mp_zh_jieba_{vocab}"
    if lang == "ja":
        return ROOT / f"tokenizer_assets/mp_ja_mecab_{vocab}"
    raise ValueError(lang)


def unigram_path(lang, vocab):
    if lang == "en":
        return (
            ROOT
            / f"tokenizer_assets/spm_en_unigram_{vocab}_5m/tokenizer.model"
        )

    if lang == "fr":
        return (
            ROOT
            / (
                "tokenizer_assets/"
                f"spm_fr_langaware_unigram_{vocab}_5m_fixednorm_portable/"
                "tokenizer.json"
            )
        )

    if lang == "zh":
        return (
            ROOT
            / f"tokenizer_assets/spm_zh_unigram_{vocab}_5m/tokenizer.model"
        )

    if lang == "ja":
        return (
            ROOT
            / f"tokenizer_assets/spm_ja_unigram_{vocab}_5m/tokenizer.model"
        )

    raise ValueError(lang)


# ============================================================
# Worker initialization
# ============================================================

def init_worker(lang):
    global _LANG
    global _WORD_VOCABS
    global _SPM
    global _EN_TOKENIZER
    global _FR_MODULE
    global _JA_TAGGER

    _LANG = lang

    # Word vocabularies
    for vocab in VOCABS:
        path = word_dir(lang, vocab) / "vocab.json"

        with open(path, encoding="utf-8") as f:
            vocab_obj = json.load(f)

        _WORD_VOCABS[vocab] = set(vocab_obj.keys())

    # Unigram tokenizers
    for vocab in VOCABS:
        path = unigram_path(lang, vocab)

        if not path.exists():
            raise FileNotFoundError(path)

        _SPM[vocab] = SubwordBackend(path)

    # English: use the actual tokenizer implementation
    if lang == "en":
        mod = load_python_module(
            word_dir("en", "10k") / "mp_tokenizer.py",
            "mp_en_analysis",
        )
        _EN_TOKENIZER = mod.MPTokenizer()

    # French: use the actual normalization/segmentation implementation
    elif lang == "fr":
        _FR_MODULE = load_python_module(
            word_dir("fr", "10k") / "mp_tokenizer.py",
            "mp_fr_analysis",
        )

    # Japanese: exact dictionary used during pretraining
    elif lang == "ja":
        from fugashi import Tagger

        import unidic_lite

        dicdir = os.environ.get(
            "MECAB_DICDIR",
            unidic_lite.DICDIR,
        )
        _JA_TAGGER = Tagger(f"-d {dicdir}")

    elif lang == "zh":
        import jieba

        jieba.initialize()


# ============================================================
# Matched preprocessing
# ============================================================

def normalize_for_subword(text):
    if text is None:
        text = ""

    text = str(text)

    if _LANG == "en":
        return text.lower()

    if _LANG == "fr":
        return _FR_MODULE.normalize_common(text)

    return text


# ============================================================
# Language-specific word-like segmentation
# ============================================================

def segment_units(text):
    if text is None:
        text = ""

    text = str(text)

    # Exact English MP tokenizer segmentation
    if _LANG == "en":
        return list(_EN_TOKENIZER.basic_tokenize(text))

    # Exact French language-aware segmentation
    if _LANG == "fr":
        return list(_FR_MODULE.segment_units(text))

    # Exact Chinese jieba segmentation used by tokenizer
    if _LANG == "zh":
        import jieba

        out = []

        for m in WS_RE.finditer(text):
            chunk = m.group(0)

            if chunk.isspace():
                out.extend(["_"] * len(chunk))
                continue

            out.extend(jieba.cut(chunk, cut_all=False))

        return out

    # Exact Japanese MeCab/fugashi segmentation used by tokenizer
    if _LANG == "ja":
        out = []

        for m in WS_RE.finditer(text):
            chunk = m.group(0)

            if chunk.isspace():
                out.extend(["_"] * len(chunk))
                continue

            for word in _JA_TAGGER(chunk):
                out.append(word.surface)

        return out

    raise ValueError(_LANG)


def is_lexical(unit):
    if unit == "_":
        return False

    return any(
        ch.isalnum()
        or unicodedata.category(ch).startswith("L")
        for ch in unit
    )


# ============================================================
# Document analysis
# ============================================================

def analyze_document(item):
    idx, text, fragmentation_docs = item

    if text is None:
        text = ""

    text = str(text)

    normalized = normalize_for_subword(text)
    units = segment_units(text)

    lexical_units = [u for u in units if is_lexical(u)]

    common = {
        "docs": 1,
        "chars": len(normalized),
        "utf8_bytes": len(normalized.encode("utf-8")),
        "units": len(units),
        "lexical_units": len(lexical_units),
    }

    conditions = {}

    # --------------------------------------------------------
    # Word-level
    # --------------------------------------------------------

    for vocab_label in VOCABS:
        vocab = _WORD_VOCABS[vocab_label]

        tokens = 0
        byte_tokens = 0

        oov_units = 0
        oov_lexical_units = 0

        for u in units:
            if u in vocab:
                tokens += 1
            else:
                nbytes = len(u.encode("utf-8"))

                tokens += nbytes
                byte_tokens += nbytes
                oov_units += 1

                if is_lexical(u):
                    oov_lexical_units += 1

        conditions[f"word_{vocab_label}"] = {
            "tokens": tokens,
            "byte_tokens": byte_tokens,
            "oov_units": oov_units,
            "oov_lexical_units": oov_lexical_units,
        }

    # --------------------------------------------------------
    # SentencePiece Unigram
    # --------------------------------------------------------

    for vocab_label in VOCABS:
        pieces = _SPM[vocab_label].pieces(normalized)

        conditions[f"unigram_{vocab_label}"] = {
            "tokens": len(pieces),
            "byte_tokens": sum(
                1 for p in pieces if BYTE_RE.fullmatch(p)
            ),
            "oov_units": 0,
            "oov_lexical_units": 0,
        }

    # --------------------------------------------------------
    # Fragmentation sample
    # --------------------------------------------------------

    fragmentation = {}

    if idx < fragmentation_docs:
        for vocab_label in VOCABS:
            fragmented = 0
            total_pieces = 0

            for u in lexical_units:
                pieces = _SPM[vocab_label].pieces(u)

                n = len(pieces)
                total_pieces += n

                if n > 1:
                    fragmented += 1

            fragmentation[vocab_label] = {
                "lexical_units": len(lexical_units),
                "fragmented_units": fragmented,
                "pieces": total_pieces,
            }

    return {
        "common": common,
        "conditions": conditions,
        "fragmentation": fragmentation,
    }


# ============================================================
# Aggregation
# ============================================================

def empty_stats():
    return {
        "docs": 0,
        "chars": 0,
        "utf8_bytes": 0,
        "units": 0,
        "lexical_units": 0,
        "tokens": 0,
        "byte_tokens": 0,
        "oov_units": 0,
        "oov_lexical_units": 0,
    }


def add_result(stats, common, cond):
    stats["docs"] += common["docs"]
    stats["chars"] += common["chars"]
    stats["utf8_bytes"] += common["utf8_bytes"]
    stats["units"] += common["units"]
    stats["lexical_units"] += common["lexical_units"]

    stats["tokens"] += cond["tokens"]
    stats["byte_tokens"] += cond["byte_tokens"]

    stats["oov_units"] += cond["oov_units"]
    stats["oov_lexical_units"] += cond["oov_lexical_units"]


def make_row(
    lang,
    cond_name,
    target_checkpoint,
    stats,
):
    tokenizer_type, vocab = cond_name.split("_", 1)

    tokens = stats["tokens"]
    units = stats["units"]
    lexical_units = stats["lexical_units"]

    row = {
        "language": lang,
        "tokenizer_type": tokenizer_type,
        "vocab_size": vocab,

        "checkpoint_tokens": target_checkpoint,
        "actual_tokens": tokens,

        "docs_covered": stats["docs"],
        "chars_covered": stats["chars"],
        "utf8_bytes_covered": stats["utf8_bytes"],

        "units_covered": units,
        "lexical_units_covered": lexical_units,

        "byte_tokens": stats["byte_tokens"],
        "byte_rate": (
            stats["byte_tokens"] / tokens if tokens else 0.0
        ),

        "tokens_per_unit": (
            tokens / units if units else 0.0
        ),

        "tokens_per_lexical_unit": (
            tokens / lexical_units if lexical_units else 0.0
        ),

        "chars_per_token": (
            stats["chars"] / tokens if tokens else 0.0
        ),

        "bytes_per_token": (
            stats["utf8_bytes"] / tokens if tokens else 0.0
        ),

        "oov_units": (
            stats["oov_units"]
            if tokenizer_type == "word"
            else None
        ),

        "oov_unit_rate": (
            stats["oov_units"] / units
            if tokenizer_type == "word" and units
            else None
        ),

        "oov_lexical_units": (
            stats["oov_lexical_units"]
            if tokenizer_type == "word"
            else None
        ),

        "oov_lexical_unit_rate": (
            stats["oov_lexical_units"] / lexical_units
            if tokenizer_type == "word" and lexical_units
            else None
        ),

        "fragmented_lexical_unit_rate": None,
        "pieces_per_lexical_unit": None,
    }

    return row


# ============================================================
# Dataset information
# ============================================================

def dataset_config(lang):
    if lang == "en":
        return {
            "dataset": "HuggingFaceTB/smollm-corpus",
            "subset": "fineweb-edu-dedup",
        }

    if lang == "fr":
        return {
            "dataset": "HuggingFaceFW/fineweb-2",
            "subset": "fra_Latn",
        }

    if lang == "zh":
        return {
            "dataset": "HuggingFaceFW/fineweb-2",
            "subset": "cmn_Hani",
        }

    if lang == "ja":
        return {
            "dataset": "hotchpotch/fineweb-2-edu-japanese",
            "subset": None,
        }

    raise ValueError(lang)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--lang",
        required=True,
        choices=["en", "fr", "zh", "ja"],
    )

    ap.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        required=True,
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--fragmentation_docs",
        type=int,
        default=2000,
        help=(
            "Number of initial documents used for "
            "Unigram fragmentation analysis."
        ),
    )

    ap.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Useful for smoke tests.",
    )

    ap.add_argument(
        "--log_every",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    args = ap.parse_args()

    checkpoints = sorted(args.checkpoints)

    conditions = [
        f"{typ}_{vocab}"
        for typ in ("word", "unigram")
        for vocab in VOCABS
    ]

    stats = {
        cond: empty_stats()
        for cond in conditions
    }

    next_checkpoint = {
        cond: 0
        for cond in conditions
    }

    rows = []

    fragmentation = {
        vocab: {
            "lexical_units": 0,
            "fragmented_units": 0,
            "pieces": 0,
        }
        for vocab in VOCABS
    }

    cfg = dataset_config(args.lang)

    print("=" * 70)
    print(f"language = {args.lang}")
    print(f"dataset  = {cfg['dataset']}")
    print(f"subset   = {cfg['subset']}")
    print(f"checkpoints = {checkpoints}")
    print("=" * 70)

    if cfg["subset"] is None:
        ds = load_dataset(
            cfg["dataset"],
            split="train",
            streaming=True,
        )
    else:
        ds = load_dataset(
            cfg["dataset"],
            cfg["subset"],
            split="train",
            streaming=True,
        )

    def all_done():
        return all(
            next_checkpoint[c] >= len(checkpoints)
            for c in conditions
        )

    def process_result(result):
        common = result["common"]

        # fragmentation statistics are independent
        # of token-budget checkpoint.
        for vocab, x in result["fragmentation"].items():
            fragmentation[vocab]["lexical_units"] += x["lexical_units"]
            fragmentation[vocab]["fragmented_units"] += x["fragmented_units"]
            fragmentation[vocab]["pieces"] += x["pieces"]

        for cond in conditions:
            if next_checkpoint[cond] >= len(checkpoints):
                continue

            add_result(
                stats[cond],
                common,
                result["conditions"][cond],
            )

            while (
                next_checkpoint[cond] < len(checkpoints)
                and stats[cond]["tokens"]
                >= checkpoints[next_checkpoint[cond]]
            ):
                target = checkpoints[next_checkpoint[cond]]

                rows.append(
                    make_row(
                        args.lang,
                        cond,
                        target,
                        stats[cond],
                    )
                )

                next_checkpoint[cond] += 1

    docs_seen = 0
    batch = []

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(args.lang,),
        mp_context=mp.get_context("spawn"),
    ) as pool:

        for example in ds:
            text = example.get("text", "")

            batch.append(
                (
                    docs_seen,
                    text,
                    args.fragmentation_docs,
                )
            )

            docs_seen += 1

            if (
                args.max_docs is not None
                and docs_seen >= args.max_docs
            ):
                pass

            if len(batch) >= args.batch_size:
                for result in pool.map(
                    analyze_document,
                    batch,
                    chunksize=4,
                ):
                    process_result(result)

                batch = []

                if docs_seen % args.log_every < args.batch_size:
                    status = ", ".join(
                        f"{c}={stats[c]['tokens'] / 1e9:.3f}B"
                        for c in conditions
                    )
                    print(
                        f"[docs={docs_seen:,}] {status}",
                        flush=True,
                    )

                if all_done():
                    break

                if (
                    args.max_docs is not None
                    and docs_seen >= args.max_docs
                ):
                    break

        if batch and not all_done():
            for result in pool.map(
                analyze_document,
                batch,
                chunksize=4,
            ):
                process_result(result)

    # --------------------------------------------------------
    # Add fragmentation sample results
    # --------------------------------------------------------

    for row in rows:
        if row["tokenizer_type"] != "unigram":
            continue

        vocab = row["vocab_size"]
        f = fragmentation[vocab]

        n = f["lexical_units"]

        row["fragmented_lexical_unit_rate"] = (
            f["fragmented_units"] / n
            if n
            else None
        )

        row["pieces_per_lexical_unit"] = (
            f["pieces"] / n
            if n
            else None
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    outbase = Path(args.output)
    outbase.parent.mkdir(parents=True, exist_ok=True)

    json_path = outbase.with_suffix(".json")
    csv_path = outbase.with_suffix(".csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with open(
            csv_path,
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print()
    print("=" * 70)
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("=" * 70)

    for row in rows:
        print(
            row["language"],
            row["tokenizer_type"],
            row["vocab_size"],
            f"budget={row['checkpoint_tokens'] / 1e9:.1f}B",
            f"docs={row['docs_covered']:,}",
            f"tok/unit={row['tokens_per_unit']:.4f}",
            f"char/tok={row['chars_per_token']:.4f}",
            f"byte/tok={row['bytes_per_token']:.4f}",
            f"byte_rate={row['byte_rate']:.4%}",
            f"oov={row['oov_lexical_unit_rate']}",
            f"frag={row['fragmented_lexical_unit_rate']}",
        )


if __name__ == "__main__":
    main()
