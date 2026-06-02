import matplotlib.pyplot as plt
import numpy as np

# =========================
# データ
# =========================
oov_counts = np.array([0, 1, 2, 3, 4])

acc_135m = np.array([0.380, 0.315, 0.294, 0.265, 0.256])
acc_360m = np.array([0.368, 0.322, 0.297, 0.275, 0.260])

# =========================
# プロット
# =========================
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(oov_counts, acc_135m, marker="o", linewidth=2, label="SmolLM2-135M")
ax.plot(oov_counts, acc_360m, marker="o", linewidth=2, label="SmolLM2-360M")

# 軸ラベル
ax.set_xlabel("Number of OOV words in prompt")
ax.set_ylabel("Accuracy")

# x軸を整数だけに
ax.set_xticks(oov_counts)

# y軸範囲
ax.set_ylim(0.24, 0.40)

# グリッド
ax.grid(True, alpha=0.3)

# 凡例
ax.legend()

# タイトル
ax.set_title("Accuracy vs. Number of OOV Words")

plt.tight_layout()

# 保存
plt.savefig("artifacts/oov_count_vs_accuracy_135m_360m.png", dpi=300, bbox_inches="tight")
plt.savefig("artifacts/oov_count_vs_accuracy_135m_360m.pdf", bbox_inches="tight")

print("Saved:")
print("artifacts/oov_count_vs_accuracy_135m_360m.png")
print("artifacts/oov_count_vs_accuracy_135m_360m.pdf")