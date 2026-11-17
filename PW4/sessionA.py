import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/concrete.csv")
print("duplicates:", df.duplicated().sum(), "| missing:", df.isna().sum().sum())
df = df.drop_duplicates().reset_index(drop=True)

print(df.describe())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(df["cement"], df["strength"], s=8)
axes[0].set_xlabel("cement (kg/m3)"); axes[0].set_ylabel("strength (MPa)")
axes[1].scatter(df["age"], df["strength"], s=8)
axes[1].set_xlabel("age (days)")
fig.savefig("results/A2_scatter.png", dpi=150, bbox_inches="tight")
X = df.drop(columns="strength")
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), len(X_test))

def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

lin = LinearRegression().fit(X_train, y_train)
pred_tr, pred_te = lin.predict(X_train), lin.predict(X_test)
print(dict(zip(X.columns, lin.coef_)), lin.intercept_)

r2_tr = r2_score(y_train, pred_tr)
print("train RMSE", np.sqrt(mean_squared_error(y_train, pred_tr)))
print("test RMSE ", np.sqrt(mean_squared_error(y_test, pred_te)))
print("test MAE  ", mean_absolute_error(y_test, pred_te))
print("train R2", r2_tr, "adj", adj_r2(r2_tr, len(X_train), X.shape[1]), "test R2", r2_score(y_test, pred_te))
resid = y_test - pred_te
fig, ax = plt.subplots()
ax.scatter(X_test["age"], resid, s=8)
ax.axhline(0, color="k", lw=1)
ax.set_xscale("log")
ax.set_xlabel("age (days, log scale)"); ax.set_ylabel("residual (MPa)")
fig.savefig("results/A4_residuals.png", dpi=150, bbox_inches="tight")

rows = [{"model": "linear", "train_rmse": np.sqrt(mean_squared_error(y_train, pred_tr)), "test_rmse": np.sqrt(mean_squared_error(y_test, pred_te)), "test_mae": mean_absolute_error(y_test, pred_te), "train_r2": r2_tr, "train_adj_r2": adj_r2(r2_tr, len(X_train), X.shape[1]), "test_r2": r2_score(y_test, pred_te)}]

terms = " ".join(f"{c:+.4f}*{n}" for n, c in zip(X.columns, lin.coef_))
print(f"strength = {lin.intercept_:.2f} {terms}   [MPa; ingredients in kg/m3, age in days]")

poly_rows = []
for d in [1, 2, 3, 4]:
    m = make_pipeline(PolynomialFeatures(degree=d, include_bias=False), StandardScaler(), LinearRegression())
    m.fit(X_train, y_train)
    p = m[0].n_output_features_
    ptr, pte = m.predict(X_train), m.predict(X_test)
    r2tr = r2_score(y_train, ptr)
    poly_rows.append({"degree": d, "p": p, "train_rmse": np.sqrt(mean_squared_error(y_train, ptr)),"test_rmse":  np.sqrt(mean_squared_error(y_test, pte)), "train_r2": r2tr, "train_adj_r2": adj_r2(r2tr, len(X_train), p), "test_r2": r2_score(y_test, pte)})
print(pd.DataFrame(poly_rows).round(3))

y_class = (df["strength"] >= 40).astype(int)
print(y_class.value_counts(normalize=True))

Xc_train, Xc_test, c_train, c_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)

knn_rows = []
for K in [1, 3, 5, 7, 15, 25, 51, 101]:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    knn.fit(Xc_train, c_train)
    knn_rows.append({"K": K, "train_acc": knn.score(Xc_train, c_train), "test_acc": knn.score(Xc_test, c_test)})
print(pd.DataFrame(knn_rows).round(3))

K_best = 7  

kdf = pd.DataFrame(knn_rows)
fig, ax = plt.subplots()
ax.plot(kdf["K"], kdf["train_acc"], "o-", label="train")
ax.plot(kdf["K"], kdf["test_acc"], "o-", label="test")
ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("accuracy"); ax.legend()
fig.savefig("results/A6_knn_K.png", dpi=150, bbox_inches="tight")

raw = KNeighborsClassifier(n_neighbors=K_best).fit(Xc_train, c_train)
sc  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_best)).fit(Xc_train, c_train)
print("test acc without scaler:", raw.score(Xc_test, c_test))
print("test acc with scaler:   ", sc.score(Xc_test, c_test))

accs = []
for rs in range(5):
    Xa, Xb, ca, cb = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_best)).fit(Xa, ca)
    accs.append(m.score(Xb, cb))
print(np.round(accs, 3), "mean", np.mean(accs).round(3), "std", np.std(accs).round(3))

tree_rows = []
for md in [1, 2, 3, 4, 5, 6, 8, None]:
    t = DecisionTreeClassifier(criterion="entropy", max_depth=md, random_state=0)
    t.fit(Xc_train, c_train)
    tree_rows.append({"max_depth": "None" if md is None else md, "leaves": t.get_n_leaves(), "train_acc": t.score(Xc_train, c_train), "test_acc": t.score(Xc_test, c_test)})
tdf = pd.DataFrame(tree_rows)
print(tdf.round(3))

full = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(Xc_train, c_train)
xs = [full.get_depth() if r["max_depth"] == "None" else r["max_depth"] for r in tree_rows]
fig, ax = plt.subplots()
ax.plot(xs, tdf["train_acc"], "o-", label="train")
ax.plot(xs, tdf["test_acc"], "o-", label="test")
ax.set_xlabel("max_depth (last point = unlimited)"); ax.set_ylabel("accuracy"); ax.legend()
fig.savefig("results/A6_tree_depth.png", dpi=150, bbox_inches="tight")

D_best = 4
tree = DecisionTreeClassifier(criterion="entropy", max_depth=D_best, random_state=0).fit(Xc_train, c_train)
print(export_text(tree, feature_names=list(X.columns), max_depth=2))

for md in [D_best, None]:
    accs = []
    for rs in range(5):
        Xa, Xb, ca, cb = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
        t = DecisionTreeClassifier(criterion="entropy", max_depth=md, random_state=0).fit(Xa, ca)
        accs.append(t.score(Xb, cb))
    print(md, np.round(accs, 3), "mean", np.mean(accs).round(3), "std", np.std(accs).round(3))

K_best, D_best = 7, 5  
knn  = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_best)).fit(Xc_train, c_train)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=D_best, random_state=0).fit(Xc_train, c_train)

def metrics_row(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "accuracy": accuracy_score(y_true, y_pred), "precision": precision_score(y_true, y_pred, zero_division=0), "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred, zero_division=0)}

final = pd.DataFrame([metrics_row("knn",  c_test, knn.predict(Xc_test)), metrics_row("tree", c_test, tree.predict(Xc_test))])
print(final.round(3))
final.to_csv("results/A_metrics.csv", index=False)

proba = knn.predict_proba(Xc_test)[:, 1]
thr = pd.DataFrame([metrics_row(f"knn t={t}", c_test, (proba >= t).astype(int)) for t in [0.3, 0.5, 0.7, 0.9]])
print(thr.round(3))

lin_std = make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)

pk = permutation_importance(knn,  Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
pt = permutation_importance(tree, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)

imp = pd.DataFrame({"raw_coef":  lin.coef_, "std_coef":  lin_std[-1].coef_, "tree_imp":  tree.feature_importances_, "perm_knn":  pk.importances_mean, "perm_knn_sd": pk.importances_std, "perm_tree": pt.importances_mean, "perm_tree_sd": pt.importances_std,}, index=X.columns)
print(imp.round(3))
imp.to_csv("results/A8_importance.csv")

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, col, title in zip(axes, ["std_coef", "tree_imp", "perm_knn"], ["standardised coef (MPa/SD)", "tree importance", "permutation, KNN (F1 drop)"]):
    imp[col].sort_values().plot.barh(ax=ax); ax.set_title(title)
fig.savefig("results/A8_importance.png", dpi=150, bbox_inches="tight")

def make_feats(D, log_age=False, wb=False):
    D = D.copy()
    if log_age:
        D["age"] = np.log(D["age"])
    if wb:
        D["wb"] = D["water"] / (D["cement"] + D["slag"] + D["ash"])
        D = D.drop(columns="water")
    return D

bonus_rows = []
for name, kw in [("baseline", {}), ("log(age)", {"log_age": True}), ("log(age) + w/b", {"log_age": True, "wb": True})]:
    Xtr, Xte = make_feats(X_train, **kw), make_feats(X_test, **kw)
    m = LinearRegression().fit(Xtr, y_train)
    bonus_rows.append({"features": name, "p": Xtr.shape[1], "train_rmse": np.sqrt(mean_squared_error(y_train, m.predict(Xtr))), "test_rmse":  np.sqrt(mean_squared_error(y_test,  m.predict(Xte))), "test_r2": r2_score(y_test, m.predict(Xte))})
bdf = pd.DataFrame(bonus_rows)
print(bdf.round(3))
bdf.to_csv("results/A_bonus.csv", index=False)
pd.DataFrame(rows).to_csv("results/A4_linear.csv", index=False)
pd.DataFrame(poly_rows).to_csv("results/A5_poly.csv", index=False)
kdf.to_csv("results/A6_knn.csv", index=False)
tdf.to_csv("results/A6_tree.csv", index=False)
thr.to_csv("results/A7_thresholds.csv", index=False)