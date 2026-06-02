import numpy as np
from scipy.stats import wilcoxon

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

def report(name, values):
    d = np.array(values, dtype=float)

    print("=" * 60)
    print(name)
    print("values =", d.tolist())
    print("n =", len(d))
    print("mean =", d.mean())
    print("std =", d.std(ddof=1))
    print("median =", np.median(d))

    # one-sided test: median(MP - SPM) > 0
    res = wilcoxon(d, alternative="greater", zero_method="wilcox", method="auto")
    print("Wilcoxon W =", res.statistic)
    print("p =", res.pvalue)

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
