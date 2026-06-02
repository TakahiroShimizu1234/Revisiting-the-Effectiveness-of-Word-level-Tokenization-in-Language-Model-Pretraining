from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tokenizer.mp_tokenizer import MPTokenizer


def build_demo_vocab():
    vocab = {
        "<s>": 0,
        "</s>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "_": 4,
        "cats": 5,
        "sit": 6,
        "on": 7,
        "100": 8,
        "1": 9,
        "0": 10,
        "apples": 11,
        "l": 12,
        "étudiant": 13,
        "étudie": 14,
        "à": 15,
        "université": 16,
        ".": 17,
        "'": 18,
    }

    next_id = len(vocab)
    for byte in range(256):
        tok = f"<0x{byte:02X}>"
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1

    return vocab


def show(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded = tokenizer.convert_tokens_to_string(tokens)

    print("=" * 80)
    print(f"Input : {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")
    print(f"Decode: {decoded}")


def main():
    tokenizer = MPTokenizer(vocab=build_demo_vocab(), lowercase=True)

    examples = [
        "cats sit on sofa",
        "L'étudiant étudie à l'université.",
        "100 apples",
        "Massachusetts NaCl ATP euglena 100%!",
    ]

    for text in examples:
        show(tokenizer, text)


if __name__ == "__main__":
    main()
