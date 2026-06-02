import json
import glob
import re
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--samples_dir")
parser.add_argument("--vocab")
args = parser.parse_args()

with open(args.vocab) as f:
    vocab = json.load(f)

vocab_set = set(vocab.keys())

stats = defaultdict(lambda: {"correct":0, "incorrect":0})

files = glob.glob(f"{args.samples_dir}/**/*.jsonl", recursive=True)

for f in files:
    with open(f) as fh:
        for line in fh:
            j = json.loads(line)

            text = j["arguments"]["gen_args_0"]["arg_0"]

            # 2文字以上単語
            words = re.findall(r"[A-Za-z]{2,}", text.lower())

            oov_count = sum(1 for w in words if w not in vocab_set)

            correct = (j["acc"] == 1)

            if correct:
                stats[oov_count]["correct"] += 1
            else:
                stats[oov_count]["incorrect"] += 1

print("\n=== OOV count vs accuracy ===\n")

for k in sorted(stats.keys()):

    correct = stats[k]["correct"]
    incorrect = stats[k]["incorrect"]
    total = correct + incorrect

    if total == 0:
        continue

    acc = correct / total

    print(
        f"OOV={k:2d} | "
        f"samples={total:5d} | "
        f"accuracy={acc:.3f}"
    )