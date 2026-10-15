import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

disease = df.loc[df["target"] == 1, "thalach"]
healthy = df.loc[df["target"] == 0, "thalach"]
print(f"disease: n={len(disease)}, mean={disease.mean()}")
print(f"healthy: n={len(healthy)}, mean={healthy.mean()}")

t, p_t = stats.ttest_ind(disease, healthy, equal_var=False)
print(f"Welch t-test:  t = {t}, p = {p_t}")
u, p_u = stats.mannwhitneyu(disease, healthy, alternative="two-sided")
print(f"Mann-Whitney:  U = {u}, p = {p_u}")

labels, means, sems, cis = ["disease", "healthy"], [], [], []
for g in (disease, healthy):
    m = g.mean()
    sem = stats.sem(g)
    lo, hi = stats.t.interval(0.95, len(g) - 1, loc=m, scale=sem)
    means.append(m); sems.append(sem); cis.append((lo, hi))
    print(f"mean = {m}, SEM = {sem}, 95% CI = ({lo}, {hi})")

plt.figure()
plt.errorbar(labels, means, yerr=[1.96 * s for s in sems], fmt="o", capsize=8, markersize=8)
plt.ylabel("max heart rate (thalach)")
plt.title("Mean thalach with 95% CI")
plt.savefig("thalach_means.png")
plt.close()

r, p_r = stats.pearsonr(df["age"], df["thalach"])
rho, p_rho = stats.spearmanr(df["age"], df["thalach"])
print(f"\nPearson r = {r} (p = {p_r})")
print(f"Spearman rho = {rho} (p = {p_rho})")

slope, intercept = np.polyfit(df["age"], df["thalach"], 1)
xs = np.linspace(df["age"].min(), df["age"].max(), 100)
plt.figure()
plt.scatter(df["age"], df["thalach"], alpha=0.6)
plt.plot(xs, slope * xs + intercept, "r-", label=f"fit: slope = {slope}")
plt.xlabel("age"); plt.ylabel("thalach"); plt.legend()
plt.savefig("age_thalach.png")
plt.close()

p = df["target"].value_counts(normalize=True)
print("\nProportions:\t", p)
entropy = -np.sum(p * np.log2(p))
print(f"Entropy of target = {entropy} bits (1.0 = perfectly balanced coin flip)")