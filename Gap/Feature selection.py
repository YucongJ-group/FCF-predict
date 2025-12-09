import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
X = df.drop(["smiles", "Gap"], axis=1)
y = df["Gap"]

pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler", StandardScaler()),
    ("rfecv", RFECV(
        estimator=RandomForestRegressor(n_estimators=100, random_state=0),
        step=1,
        cv=KFold(5, shuffle=True, random_state=0),
        scoring="r2",
        n_jobs=-1
    ))
])

pipeline.fit(X, y)
rfecv = pipeline.named_steps["rfecv"]
print(rfecv.n_features_)
selected_mask = rfecv.support_
selected_features = X.columns[selected_mask].tolist()
print(selected_features)

df_out = pd.DataFrame({
    "feature": X.columns,
    "selected": selected_mask.astype(int)
})
df_out["ranking"] = rfecv.ranking_
df_out.to_csv("Gap_rfecv_selected.csv", index=False)
print("Gap_rfecv_selected.csv")

cv_results = pd.DataFrame({
    "n_features": rfecv.cv_results_["n_features"],
    "mean_r2": rfecv.cv_results_["mean_test_score"],
    "std_r2": rfecv.cv_results_["std_test_score"]
})
plt.figure(figsize=(6,4))
plt.plot(cv_results["n_features"], cv_results["mean_r2"], '-o')
plt.fill_between(
    cv_results["n_features"],
    cv_results["mean_r2"] - cv_results["std_r2"],
    cv_results["mean_r2"] + cv_results["std_r2"], alpha=0.2
)
plt.xlabel("Number of features selected")
plt.ylabel("Cross-Validated R²")
plt.title("RFECV: Feature # vs CV R²")
plt.show()
