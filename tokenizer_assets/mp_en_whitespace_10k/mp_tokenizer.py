
import json
from pathlib import Path

class MPTokenizer:
    def __init__(self, vocab_file=None):
        if vocab_file is None:
            vocab_file = Path(__file__).with_name("vocab.json")
        with open(vocab_file, encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token_id = self.vocab[self.bos_token]
        self.eos_token_id = self.vocab[self.eos_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

    def __len__(self):
        return len(self.vocab)

    def basic_tokenize(self, text):
        parts = text.split()
        for i, tok in enumerate(parts):
            if i > 0:
                yield "_"
            yield tok

    def token_to_ids(self, tok):
        if tok in self.vocab:
            return [self.vocab[tok]]
        out = []
        for b in tok.encode("utf-8", errors="replace"):
            out.append(self.vocab.get(f"<0x{b:02X}>", self.unk_token_id))
        return out

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        for tok in self.basic_tokenize(text):
            ids.extend(self.token_to_ids(tok))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(int(i), self.unk_token) for i in ids]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        toks = self.convert_ids_to_tokens(ids)
        out = bytearray()
        text_parts = []

        def flush_bytes():
            nonlocal out
            if out:
                text_parts.append(out.decode("utf-8", errors="replace"))
                out = bytearray()

        for tok in toks:
            if skip_special_tokens and tok in {self.bos_token, self.eos_token, self.pad_token, self.unk_token}:
                continue
            if tok == "_":
                flush_bytes()
                text_parts.append(" ")
            elif tok.startswith("<0x") and tok.endswith(">"):
                try:
                    out.append(int(tok[3:5], 16))
                except Exception:
                    flush_bytes()
                    text_parts.append(tok)
            else:
                flush_bytes()
                text_parts.append(tok)
        flush_bytes()
        return "".join(text_parts).strip()

    def batch_decode(self, sequences, skip_special_tokens=True, **kwargs):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) for seq in sequences]


# Hugging Face Trainer checkpoint compatibility
def _mp_get_vocab(self):
    return dict(self.vocab)


def _mp_save_pretrained(self, save_directory, **kwargs):
    import json
    import shutil
    from pathlib import Path

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    vocab_path = save_directory / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(self.vocab, f, ensure_ascii=False)

    source_path = Path(__file__)
    tokenizer_code_path = save_directory / "mp_tokenizer.py"
    if source_path.resolve() != tokenizer_code_path.resolve():
        shutil.copy2(source_path, tokenizer_code_path)

    tokenizer_config_path = save_directory / "tokenizer_config.json"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "MPTokenizer",
                "model_max_length": 8192,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    special_tokens_path = save_directory / "special_tokens_map.json"
    with open(special_tokens_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return (
        str(vocab_path),
        str(tokenizer_code_path),
        str(tokenizer_config_path),
        str(special_tokens_path),
    )


MPTokenizer.get_vocab = _mp_get_vocab
MPTokenizer.save_pretrained = _mp_save_pretrained
