import json
import re
import unicodedata
from pathlib import Path

_APOSTROPHE_TRANSLATION = str.maketrans({
    "’": "'",
    "‘": "'",
    "ʼ": "'",
    "＇": "'",
    "`": "'",
    "´": "'",
})

_TOKEN_PATTERN = re.compile(
    r"[^\W\d_]+(?:'[^\W\d_]+)*|\d+|\s+|[^\s]",
    re.UNICODE,
)

_CLITIC_PATTERN = re.compile(
    r"^(qu|[cdjlmnst])'(.+)$",
    re.UNICODE,
)

_APOSTROPHE_EXCEPTIONS = {
    "aujourd'hui",
    "prud'homme",
    "presqu'île",
    "quelqu'un",
    "quelqu'une",
}


def normalize_common(text):
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_APOSTROPHE_TRANSLATION)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def segment_units(text):
    text = normalize_common(text)

    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)

        if token.isspace():
            for _ in token:
                yield "_"
            continue

        if token.isdigit():
            yield from token
            continue

        if "'" in token and token not in _APOSTROPHE_EXCEPTIONS:
            clitic = _CLITIC_PATTERN.match(token)
            if clitic:
                yield clitic.group(1) + "'"
                yield clitic.group(2)
                continue

        yield token


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

        self.model_max_length = 8192
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self):
        return len(self.vocab)

    def basic_tokenize(self, text):
        yield from segment_units(text)

    def tokenize(self, text):
        return list(self.basic_tokenize(text))

    def token_to_ids(self, token):
        if token in self.vocab:
            return [self.vocab[token]]

        return [
            self.vocab.get(f"<0x{byte:02X}>", self.unk_token_id)
            for byte in token.encode("utf-8", errors="replace")
        ]

    def encode(self, text, add_special_tokens=True, **kwargs):
        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        for token in self.basic_tokenize(text):
            ids.extend(self.token_to_ids(token))

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, (list, tuple)):
            encoded = [
                self.encode(item, add_special_tokens=add_special_tokens)
                for item in text
            ]
            return {"input_ids": encoded}

        return {
            "input_ids": self.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        }

    def convert_ids_to_tokens(self, ids):
        return [
            self.id_to_token.get(int(index), self.unk_token)
            for index in ids
        ]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)

        return [
            self.vocab.get(token, self.unk_token_id)
            for token in tokens
        ]

    def get_vocab(self):
        return dict(self.vocab)

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        tokens = self.convert_ids_to_tokens(ids)
        byte_buffer = bytearray()
        output = []

        def flush_bytes():
            nonlocal byte_buffer
            if byte_buffer:
                output.append(
                    byte_buffer.decode("utf-8", errors="replace")
                )
                byte_buffer = bytearray()

        for token in tokens:
            if (
                skip_special_tokens
                and token in {
                    self.bos_token,
                    self.eos_token,
                    self.pad_token,
                    self.unk_token,
                }
            ):
                continue

            if token == "_":
                flush_bytes()
                output.append(" ")
            elif token.startswith("<0x") and token.endswith(">"):
                try:
                    byte_buffer.append(int(token[3:5], 16))
                except ValueError:
                    flush_bytes()
                    output.append(token)
            else:
                flush_bytes()
                output.append(token)

        flush_bytes()
        return "".join(output).strip()

    def batch_decode(
        self,
        sequences,
        skip_special_tokens=True,
        **kwargs,
    ):
        return [
            self.decode(
                sequence,
                skip_special_tokens=skip_special_tokens,
            )
            for sequence in sequences
        ]

    def save_pretrained(self, save_directory, **kwargs):
        import shutil

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        vocab_path = save_directory / "vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        source_path = Path(__file__)
        code_path = save_directory / "mp_tokenizer.py"

        if source_path.resolve() != code_path.resolve():
            shutil.copy2(source_path, code_path)

        config_path = save_directory / "tokenizer_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tokenizer_class": "MPTokenizer",
                    "model_max_length": self.model_max_length,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "language": "fr",
                    "tokenizer_type": "language-aware-word-level",
                    "normalization": "NFC+lowercase+apostrophe-normalization",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        special_path = save_directory / "special_tokens_map.json"
        with open(special_path, "w", encoding="utf-8") as f:
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
            str(code_path),
            str(config_path),
            str(special_path),
        )
