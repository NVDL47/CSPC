import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/concrete.csv")
print("duplicates:", df.duplicated().sum(), "| missing:", df.isna().sum().sum())
df = df.drop_duplicates().reset_index(drop=True)
X = df.drop(columns="strength")
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
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