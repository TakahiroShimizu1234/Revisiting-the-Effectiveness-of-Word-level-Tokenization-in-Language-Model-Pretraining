from pathlib import Path
import json
import re

ROOT = Path(".")

files = []
for pat in ["**/tokenizer.json", "**/vocab.json", "**/vocab.txt", "**/*.vocab"]:
    files.extend(ROOT.glob(pat))

# MP / word-level っぽいものを優先
files = [p for p in files if re.search(r"(mp|word|fwe|tokenizer_assets)", str(p), re.I)]

def load_vocab(path):
    name = path.name

    if name == "tokenizer.json":
        data = json.loads(path.read_text(errors="ignore"))
        vocab = {}

        # HuggingFace WordLevel/BPE など
        model = data.get("model", {})
        mv = model.get("vocab", None)
        if isinstance(mv, dict):
            vocab.update(mv)
        elif isinstance(mv, list):
            for i, tok in enumerate(mv):
                if isinstance(tok, str):
                    vocab[tok] = i
                elif isinstance(tok, list) and tok:
                    vocab[tok[0]] = i

        # added_tokens も一応含める
        for t in data.get("added_tokens", []):
            content = t.get("content")
            tid = t.get("id")
            if content is not None:
                vocab[content] = tid

        return vocab

    if name == "vocab.json":
        data = json.loads(path.read_text(errors="ignore"))
        if isinstance(data, dict):
            return data
        return {}

    # vocab.txt / *.vocab
    vocab = {}
    for i, line in enumerate(path.read_text(errors="ignore").splitlines()):
        if not line.strip():
            continue
        tok = line.split()[0]
        vocab[tok] = i
    return vocab

def summarize(path):
    vocab = load_vocab(path)
    if not vocab:
        return

    tokens = set(vocab.keys())

    specials = ["<s>", "</s>", "<pad>", "<unk>"]
    byte_tokens = [t for t in tokens if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", t)]

    print("=" * 100)
    print(path)
    print(f"vocab_size_total = {len(tokens)}")
    print(f"has_underscore   = {'_' in tokens}")
    print(f"num_byte_tokens  = {len(byte_tokens)}")
    print("special_tokens   =", {s: (s in tokens) for s in specials})

    reserved = sum(s in tokens for s in specials) + (1 if "_" in tokens else 0) + len(byte_tokens)
    print(f"reserved_count   = specials_present + underscore + byte_tokens = {reserved}")
    print(f"non_reserved_est = {len(tokens) - reserved}")

for p in sorted(set(files)):
    summarize(p)
