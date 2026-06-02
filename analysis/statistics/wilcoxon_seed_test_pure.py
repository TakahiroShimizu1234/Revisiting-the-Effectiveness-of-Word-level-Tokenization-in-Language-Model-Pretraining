import itertools
import math

# SmolLM2 135M, seed 42 / 2024 / 123
# Each value is word-level performance minus subword performance.
# Source: Appendix C, Table 5.

acc = {
    "10k":  [0.0221, -0.0077, -0.0021],
    "50k":  [0.0295,  0.0313, -0.0051],
    "100k": [0.0118, -0.0321, -0.0024],
}

acc_norm = {
    "10k":  [0.0104, 0.0019, 0.0015],
    "50k":  [0.0405, 0.0343, 0.0353],
    "100k": [0.0417, 0.0351, 0.0267],
}

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n % 2 == 1:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])

def average_ranks(abs_vals):
    order = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i])
    ranks = [0.0] * len(abs_vals)

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and abs_vals[order[j]] == abs_vals[order[i]]:
            j += 1

        # ranks are 1-based
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    return ranks

def wilcoxon_greater(diff):
    # Remove zeros, as in the standard Wilcoxon signed-rank test.
    d = [x for x in diff if x != 0]

    abs_vals = [abs(x) for x in d]
    ranks = average_ranks(abs_vals)

    # W+ statistic
    W_obs = sum(r for x, r in zip(d, ranks) if x > 0)

    # Exact one-sided p-value:
    # P(W+ >= observed W+) under random signs.
    all_W = []
    for signs in itertools.product([0, 1], repeat=len(ranks)):
        W = sum(r for s, r in zip(signs, ranks) if s == 1)
        all_W.append(W)

    p = sum(W >= W_obs for W in all_W) / len(all_W)
    return W_obs, p

def report(name, values):
    W, p = wilcoxon_greater(values)

    print("=" * 70)
    print(name)
    print("values =", values)
    print("n =", len(values))
    print("mean =", mean(values))
    print("std =", std(values))
    print("median =", median(values))
    print("Wilcoxon W+ =", W)
    print("one-sided p =", p)

print("\n### Per-vocabulary tests: acc ###")
for vocab, values in acc.items():
    report(f"acc {vocab}", values)

print("\n### Per-vocabulary tests: acc_norm ###")
for vocab, values in acc_norm.items():
    report(f"acc_norm {vocab}", values)

print("\n### Pooled seed-vocabulary tests ###")
acc_all = [x for values in acc.values() for x in values]
acc_norm_all = [x for values in acc_norm.values() for x in values]

report("acc pooled over seed-vocab", acc_all)
report("acc_norm pooled over seed-vocab", acc_norm_all)
