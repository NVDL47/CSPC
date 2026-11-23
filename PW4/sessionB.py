import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

os.makedirs("results", exist_ok=True)

wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target
print(wine.DESCR)
print(X.describe().T[["mean", "std", "min", "max"]])

pca_raw = PCA().fit(X)
print("PC1 explained variance ratio:", pca_raw.explained_variance_ratio_[0])
print(pd.Series(pca_raw.components_[0], index=X.columns).sort_values(key=abs, ascending=False))

scaler = StandardScaler()
Z = scaler.fit_transform(X)
print(Z.mean(axis=0).round(6), Z.std(axis=0).round(6))
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
print(pd.DataFrame({"evr": evr, "cumulative": cum}).round(3))
print("components for 90%:", int(np.argmax(cum >= 0.9)) + 1, "| PC1-PC2 keep:", cum[1])

fig, ax = plt.subplots()
ax.bar(range(1, 14), evr)
ax.plot(range(1, 14), cum, "o-", color="k")
ax.axhline(0.9, color="r", ls="--")
ax.set_xlabel("component"); ax.set_ylabel("variance ratio")
fig.savefig("results/B3_scree.png", dpi=150, bbox_inches="tight")

load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
print(load.sort_values("PC1"))
print(load.sort_values("PC2"))

scores = pca.transform(Z)
fig, ax = plt.subplots()
ax.scatter(scores[:, 0], scores[:, 1], s=15)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.savefig("results/B3_pc1_pc2.png", dpi=150, bbox_inches="tight")

inert_a = [KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=rs).fit(Z).inertia_ for rs in range(8)]
print(np.round(inert_a, 3))

ks = list(range(1, 11))
inert_b = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z).inertia_ for k in ks]
print(pd.DataFrame({"k": ks, "inertia": np.round(inert_b, 1)}))
fig, ax = plt.subplots()
ax.plot(ks, inert_b, "o-"); ax.set_xlabel("k"); ax.set_ylabel("inertia")
fig.savefig("results/B4_elbow.png", dpi=150, bbox_inches="tight")
plt.close(fig)

K_CHOSEN = 3
km = KMeans(n_clusters=K_CHOSEN, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(Z)
print("sizes:", np.bincount(km.labels_))
centres = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
print(centres.round(2).T)

fig, ax = plt.subplots()
ax.scatter(scores[:, 0], scores[:, 1], c=km.labels_, s=15)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.savefig("results/B5_clusters.png", dpi=150, bbox_inches="tight")
plt.close(fig)

def agreement(labels):
    table = pd.crosstab(cultivar, labels)
    return table, table.max(axis=0).sum() / len(cultivar)

versions = {"scaled, 13 features": Z, "raw, unscaled": X, "PC1-PC2 of scaled": scores[:, :2]}
res = []
for name, data in versions.items():
    k3 = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(data)
    table, agr = agreement(k3.labels_)
    print(name); print(table); print("agreement:", round(agr, 3), "\n")
    res.append({"version": name, "agreement": agr})
pd.DataFrame(res).to_csv("results/B_agreement.csv", index=False)

pd.DataFrame({"k": ks, "inertia": inert_b}).to_csv("results/B4_inertia.csv", index=False)
fig, ax = plt.subplots()
ax.scatter(scores[:, 0], scores[:, 1], c=cultivar, s=15)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.savefig("results/B6_cultivars.png", dpi=150, bbox_inches="tight")
plt.close(fig)