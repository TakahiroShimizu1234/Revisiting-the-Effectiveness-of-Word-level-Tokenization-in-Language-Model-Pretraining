from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--model_type", choices=["unigram", "bpe"], required=True)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--byte_fallback", action="store_true")
    parser.add_argument("--normalization_rule_name", default="identity")
    args = parser.parse_args()

    model_prefix = Path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        byte_fallback=args.byte_fallback,
        normalization_rule_name=args.normalization_rule_name,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
    )

    print(f"Saved SentencePiece model to: {model_prefix}.model")
    print(f"Saved SentencePiece vocab to: {model_prefix}.vocab")


if __name__ == "__main__":
    main()
