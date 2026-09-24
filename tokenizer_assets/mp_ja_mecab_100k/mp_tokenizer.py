from __future__ import annotations

import os

import json
import re
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer
from fugashi import Tagger

_WS_RE = re.compile(r"\s+|\S+", re.UNICODE)

def utf8_bytes_tokens(s: str) -> List[str]:
    return [f"<0x{b:02X}>" for b in s.encode("utf-8")]

class MPTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        vocab_file: Optional[str] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        **kwargs,
    ):
        if vocab is None:
            if vocab_file is None:
                raise ValueError("Either `vocab` or `vocab_file` must be provided.")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        import unidic_lite

        dicdir = os.environ.get(
            "MECAB_DICDIR",
            unidic_lite.DICDIR,
        )
        self.tagger = Tagger(f"-d {dicdir}")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        out = []
        for m in _WS_RE.finditer(text):
            chunk = m.group(0)
            if chunk.isspace():
                out.extend(["_"] * len(chunk))
                continue

            for word in self.tagger(chunk):
                tok = word.surface
                if tok in self.vocab:
                    out.append(tok)
                else:
                    out.extend(utf8_bytes_tokens(tok))
        return out

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.inv_vocab.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        out = []
        buf = bytearray()

        def flush():
            nonlocal buf
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()

        for t in tokens:
            if t == "_":
                flush()
                out.append(" ")
            elif t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                try:
                    buf.append(int(t[3:5], 16))
                except Exception:
                    flush()
                    out.append(t)
            else:
                flush()
                out.append(t)
        flush()
        return "".join(out)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = f"{save_directory.rstrip('/')}/{name}"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        return (path,)
