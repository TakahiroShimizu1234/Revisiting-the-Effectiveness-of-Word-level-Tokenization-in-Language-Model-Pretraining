# mp_tokenizer.py
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

# unit split rules:
#  - [A-Za-z]+
#  - digits \d+  -> split each digit
#  - whitespace \s+  -> handled as "_" tokens (one per char)
#  - otherwise: any single non-space char (symbol)
_UNIT_RE = re.compile(r"[A-Za-z]+|\d+|\s+|[^\s]", re.UNICODE)


def split_units(text: str) -> List[str]:
    units: List[str] = []
    for m in _UNIT_RE.finditer(text):
        s = m.group(0)
        if s.isdigit():
            units.extend(list(s))  # "100" -> "1","0","0"
        else:
            units.append(s)
    return units


def utf8_bytes_tokens(s: str) -> List[str]:
    b = s.encode("utf-8")
    return [f"<0x{byte:02X}>" for byte in b]


class MPTokenizer(PreTrainedTokenizer):
    """
    Meaning-Preserving tokenizer (no subword):
      - dictionary uses raw tokens (e.g., "a", "hello", "!", "0")
      - whitespace is represented by "_" (special-cased)
      - matching is done in lowercase (for non-whitespace)
      - known unit -> emits the raw token (e.g., "hello")
      - unknown unit -> emits UTF-8 byte tokens: <0x..>
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        vocab_file: Optional[str] = None,
        lowercase: bool = True,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        # Load vocab
        if vocab is None:
            if vocab_file is None:
                raise ValueError("Either `vocab` or `vocab_file` must be provided.")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

        self.vocab: Dict[str, int] = vocab
        self.inv_vocab: Dict[int, str] = {i: t for t, i in vocab.items()}
        self.lowercase = lowercase

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

        # Ensure HF-required special tokens exist in vocab
        self._ensure_in_vocab([bos_token, eos_token, pad_token, unk_token])

        # Ensure "_" exists (since you added it to vocab_100000.txt line 1)
        if "_" not in self.vocab:
            self._ensure_in_vocab(["_"])

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _ensure_in_vocab(self, tokens: List[str]) -> None:
        for t in tokens:
            if t not in self.vocab:
                idx = len(self.vocab)
                self.vocab[t] = idx
                self.inv_vocab[idx] = t

    def _tokenize(self, text: str) -> List[str]:
        units = split_units(text)
        out_tokens: List[str] = []

        for u in units:
            # Whitespace: represent as "_" per character
            # "  " -> ["_", "_"], "\n" -> ["_"] (if you prefer distinct tokens for \n, change here)
            if u.isspace():
                out_tokens.extend(["_"] * len(u))
                continue

            key = u.lower() if self.lowercase else u

            if key in self.vocab:
                out_tokens.append(key)
            else:
                out_tokens.extend(utf8_bytes_tokens(u))

        return out_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # decode: "_" -> space, <0x..> -> UTF-8 decode, others -> raw concat
        out_chars: List[str] = []
        buf: bytearray = bytearray()

        def flush_bytes() -> None:
            nonlocal buf
            if buf:
                out_chars.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()

        for t in tokens:
            if t == "_":
                flush_bytes()
                out_chars.append(" ")
                continue

            if t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                try:
                    buf.append(int(t[3:5], 16))
                except Exception:
                    flush_bytes()
                    out_chars.append(t)
                continue

            flush_bytes()
            out_chars.append(t)

        flush_bytes()
        return "".join(out_chars)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        # bos + seq + eos (pair: bos + a + eos + b + eos)
        if token_ids_1 is not None:
            return (
                [self.bos_token_id]
                + token_ids_0
                + [self.eos_token_id]
                + token_ids_1
                + [self.eos_token_id]
            )
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        name = (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        path = f"{save_directory.rstrip('/')}/{name}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)


if __name__ == "__main__":
    # Quick smoke test (optional)
    import sys

    if len(sys.argv) >= 2:
        vocab_path = sys.argv[1]
        tok = MPTokenizer(vocab_file=vocab_path, lowercase=True)
        s = "Massachusetts NaCl ATP euglena 100%!"
        enc = tok(s, add_special_tokens=False)
        toks = tok.convert_ids_to_tokens(enc["input_ids"])
        print("input :", s)
        print("tokens:", toks)
        print("decode:", tok.decode(enc["input_ids"], skip_special_tokens=True))
    else:
        print("Usage: python3 mp_tokenizer.py vocab.json")
