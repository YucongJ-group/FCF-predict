import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_db = pd.read_csv("database.csv")

features = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'qed',
            'HeavyAtomMolWt', 'MaxPartialCharge', 'MinPartialCharge',
            'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan3',
            'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI',
            'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'BalabanJ',
            'BertzCT', 'Chi2n', 'Chi4n', 'HallKierAlpha', 'Kappa1', 'Kappa2',
            'PEOE_VSA1', 'PEOE_VSA11', 'PEOE_VSA14', 'PEOE_VSA4', 'SMR_VSA10',
            'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SlogP_VSA1', 'SlogP_VSA5',
            'TPSA', 'EState_VSA10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
            'FractionCSP3', 'MolLogP', 'fr_C_O', 'fr_aniline', 'fr_nitrile',
            'fr_nitro']

X = df_train[features].values
y = df_train["LUMO"].values
X_test = df_test[features].values
y_test = df_test["LUMO"].values

df_db_clean = df_db.dropna(subset=features)
X_db = df_db_clean[features].values

imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

X_imp = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

X_test_imp = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imp)

X_db_imp = imputer.transform(X_db)
X_db_scaled = scaler.transform(X_db_imp)

X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

fixed_params = {
    'n_estimators': 1000,
    'max_depth': 17,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': None,
    'random_state': 42,
    'n_jobs': -1
}
model = RandomForestRegressor(**fixed_params)
model.fit(X_tr, y_tr)

val_preds = model.predict(X_val)
print("Validation R²:", r2_score(y_val, val_preds),
      "Validation MSE:", mean_squared_error(y_val, val_preds))

model.fit(X_scaled, y)

# 保存模型
joblib.dump(model, 'rf_model_LUMO_fixed.joblib')
print("Saved model 'rf_model_LUMO_fixed.joblib'")

pred_train = model.predict(X_scaled)
pred_test = model.predict(X_test_scaled)
print("Train R²:", r2_score(y, pred_train), "MSE:", mean_squared_error(y, pred_train))
print("Test  R²:", r2_score(y_test, pred_test), "MSE:", mean_squared_error(y_test, pred_test))

df_train["pred"] = pred_train
df_test["pred"] = pred_test
df_train[["smiles", "LUMO", "pred"]].to_csv("train_LUMO_fixed_rf.csv", index=False)
df_test[["smiles", "LUMO", "pred"]].to_csv("test_LUMO_fixed_rf.csv", index=False)

pred_db = model.predict(X_db_scaled)
df_db_out = df_db_clean.copy()
df_db_out['pred_LUMO'] = pred_db

cols_db_save = []
if 'smiles' in df_db_out.columns:
    cols_db_save.append('smiles')
if 'LUMO' in df_db_out.columns:
    cols_db_save.append('LUMO')
cols_db_save.append('pred_LUMO')

df_db_out.to_csv("database_LUMO_predictions.csv", index=False, columns=cols_db_save)
print("✅ database.csv results were saved as 'database_LUMO_predictions.csv'")

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
top5_feats = np.array(features)[indices][:5]
top5_imp = importance[indices][:5]

plt.figure(figsize=(10, 6))
plt.barh(top5_feats, top5_imp, color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance', fontsize=16, family='Arial', fontweight='bold')
plt.ylabel('Features', fontsize=16, family='Arial', fontweight='bold')
plt.title('Top 5 Feature Importances (Random Forest)', fontsize=20, family='Arial', fontweight='bold')
plt.xticks(fontsize=14, family='Arial', fontweight='bold')
plt.yticks(fontsize=14, family='Arial', fontweight='bold')
plt.tight_layout()
plt.savefig("rf_top5_feature_importance_fixed.png", dpi=300)
plt.show()

explainer = shap.TreeExplainer(model)
X_sample = shap.utils.sample(X_scaled, min(500, X_scaled.shape[0]))
shap_values = explainer(X_sample)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type="bar",
                  max_display=5, show=False)
ax1.set_title("Top 5 SHAP Feature Importances", fontsize=20, family='Arial', fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_xlabel("mean(|SHAP value|) (impact magnitude)", fontsize=16, family='Arial', fontweight='bold')

shap.summary_plot(shap_values, X_sample, feature_names=features, plot_type="dot",
                  max_display=5, show=False, alpha=0.7)
ax2.set_title("SHAP Value Distribution (Beeswarm Plot)", fontsize=20, family='Arial', fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_xlabel("SHAP value (impact on model output)", fontsize=16, family='Arial', fontweight='bold')

plt.tight_layout()
plt.savefig("rf_shap_combined_fixed.png", dpi=300, bbox_inches='tight')
plt.show()

X_top5 = X_scaled[:, indices[:5]]
combined_data = np.hstack([X_top5, y.reshape(-1, 1)])
combined_features = list(top5_feats) + ["LUMO"]
corr_matrix = np.corrcoef(combined_data, rowvar=False)
corr_df = pd.DataFrame(corr_matrix, index=combined_features, columns=combined_features)

plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(
    corr_df,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=1,
    annot_kws={'size': 15, 'weight':'bold', 'family':'Arial'},
    cbar_kws={'shrink': 1},
    vmin=-1, vmax=1
)
plt.title('Pearson Correlation Heatmap (Top 5 Features & LUMO)', fontsize=16, family='Arial', fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=15, family='Arial', fontweight='bold')
plt.yticks(rotation=0, fontsize=15, family='Arial', fontweight='bold')
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontfamily('Arial')
plt.tight_layout()
plt.savefig("rf_correlation_heatmap_fixed.png", dpi=300)
plt.show()
