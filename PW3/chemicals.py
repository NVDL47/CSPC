import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("chemicals_cancer.csv")
chems = ["benzene", "cadmium"]

print("Naive correlation with malignancy (all patients):")
for c in chems:
    r, p = stats.pearsonr(df[c], df["malignancy"])
    print(f"  {c} r = {r}  (p = {p})")

print("\nFull correlation matrix:")
print(df.corr().round(3))

for lo, hi in [(40, 60), (45, 55), (30, 50), (60, 80)]:
    sub = df[(df["pollution_index"] > lo) & (df["pollution_index"] < hi)]
    print(f"\nPollution band ({lo}, {hi}): n = {len(sub)}")
    for c in chems:
        r, p = stats.pearsonr(sub[c], sub["malignancy"])
        print(f"  {c} r = {r}  (p = {p})")

def residual(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)

pol = df["pollution_index"]
res_m = residual(df["malignancy"], pol)
print("\nPartial correlation (pollution_index removed):")
for c in chems:
    r, p = stats.pearsonr(residual(df[c], pol), res_m)
    print(f"  {c} r = {r}  (p = {p})")

band = df[(df["pollution_index"] > 40) & (df["pollution_index"] < 60)]
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
for j, c in enumerate(chems):
    axes[0, j].scatter(df[c], df["malignancy"], s=8, alpha=0.5)
    axes[0, j].set_title(f"{c}: all patients")
    axes[1, j].scatter(band[c], band["malignancy"], s=8, alpha=0.5)
    axes[1, j].set_title(f"{c}: pollution_index 40-60")
    axes[1, j].set_xlabel(c)
for ax in axes[:, 0]:
    ax.set_ylabel("malignancy")
plt.tight_layout()
plt.savefig("chemicals.png")