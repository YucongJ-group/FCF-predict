import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import combinations
import joblib
import numpy as np

df = pd.read_csv('DN_database.csv')
y = df['DN']

all_features = ['HOMO', 'LUMO', 'q-', 'q+', 'dipole', 'Gap']

feature_combos = []
for r in range(2, len(all_features) + 1):
    feature_combos.extend(combinations(all_features, r))

best_r = -1
best_features = None
best_weights = None
best_scaler = None

results = []

for features in feature_combos:
    print(f"\n>>> {features}")

    X = df[list(features)]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression(fit_intercept=False)
    model.fit(Xs, y)
    weights = model.coef_

    weight_str = ", ".join([f"{f}:{w:.4f}" for f, w in zip(features, weights)])
    print(f"{weight_str}")

    fused = Xs @ weights
    r, _ = pearsonr(fused, y)
    print(f"{features} Pearson r = {r:.4f}")

    new_col_name = '+'.join(features)
    df[new_col_name] = fused

    results.append({
        "features": '+'.join(features),
        "weights": weight_str,
        "pearson_r": r
    })

    if r > best_r:
        best_r = r
        best_features = features
        best_weights = weights
        best_scaler = StandardScaler()
        best_scaler.mean_ = scaler.mean_.copy()
        best_scaler.scale_ = scaler.scale_.copy()
        best_scaler.var_ = scaler.var_.copy()
        best_scaler.n_features_in_ = scaler.n_features_in_
        best_scaler.n_samples_seen_ = scaler.n_samples_seen_

if best_scaler is not None:
    joblib.dump(best_scaler, 'best_feature_scaler.pkl')
    print(f"\n✅ Scaler: best_feature_scaler.pkl")
    print(f"{best_features}")
    print(f"{best_r:.4f}")

    best_weights_info = {
        'features': list(best_features),
        'weights': best_weights.tolist(),
        'pearson_r': best_r
    }
    import json

    with open('best_weights.json', 'w') as f:
        json.dump(best_weights_info, f)
    print(f"✅ best_weights.json")

df.to_csv('DN_with_fused_features.csv', index=False)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="pearson_r", ascending=False)
results_df.to_csv('fusion_results_summary.csv', index=False)

print("\n✅ Saved file：")
print("1. DN_with_fused_features.csv")
print("2. fusion_results_summary.csv")
print("3. best_feature_scaler.pkl")
print("4. best_weights.json")