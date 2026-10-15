import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")
print(f"Shape: {df.shape}")
print(df.head())

cols = ["age", "chol", "trestbps", "thalach"]
dl = df[cols]
print(dl.describe())

fig, axes = plt.subplots(2, 2)
for ax, c in zip(axes.ravel(), cols):
    ax.hist(dl[c], bins=20, density=True, alpha=0.5)
    dl[c].plot.kde(ax=ax)
    ax.set_title(c)
    ax.set_xlabel(c)
plt.tight_layout()
plt.savefig("distributions.png")
plt.close(fig)

fig, axes = plt.subplots(2, 2)
for ax, c in zip(axes.ravel(), cols):
    stats.probplot(dl[c], dist="norm", plot=ax)
    ax.set_title(f"Q-Q plot: {c}")
plt.tight_layout()
plt.savefig("qqplots.png")
plt.close(fig)

print(f"\n{'var'} {'skew'} {'kurt'} {'shapiro W'} {'p-value'}")
for c in cols:
    skew = stats.skew(dl[c])
    kurt = stats.kurtosis(dl[c])
    W, p = stats.shapiro(dl[c])
    print(f"{c} {skew} {kurt} {W} {p}")