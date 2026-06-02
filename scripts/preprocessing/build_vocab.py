from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tokenizer.mp_tokenizer import split_units

SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>", "_"]


def byte_tokens():
    return [f"<0x{b:02X}>" for b in range(256)]


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.is_file():
            yield p


def count_units(path: Path, lowercase: bool = True):
    counter = Counter()
    for file_path in iter_files(path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                for unit in split_units(line):
                    if unit.isspace():
                        continue
                    if lowercase:
                        unit = unit.lower()
                    counter[unit] += 1
    return counter


def build_vocab(counter, vocab_size: int):
    vocab = {}

    def add(token):
        if token not in vocab:
            vocab[token] = len(vocab)

    for token in SPECIAL_TOKENS:
        add(token)

    for token in byte_tokens():
        add(token)

    for token, _ in counter.most_common():
        if len(vocab) >= vocab_size:
            break
        add(token)

    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--no_lowercase", action="store_true")
    args = parser.parse_args()

    counter = count_units(Path(args.input), lowercase=not args.no_lowercase)
    vocab = build_vocab(counter, args.vocab_size)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved vocabulary to: {output_path}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Counted units: {len(counter)}")


if __name__ == "__main__":
    main()
