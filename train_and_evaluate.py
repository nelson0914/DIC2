import itertools
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# PHASE 1 & 2: DATA INGESTION & UNDERSTANDING
# ============================================================
print("--- Starting DIC2 Regression & Feature Selection Pipeline ---")

url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
try:
    df = pd.read_csv(url)
except Exception:
    # Fallback local if offline
    df = pd.read_csv("50_Startups.csv")

# Save raw dataset locally for the web app explorer
df.to_csv("50_Startups.csv", index=False)

# One-hot encode State (California dropped as reference category)
df_enc = pd.get_dummies(df, columns=["State"], dtype=int)

features = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State_Florida",
    "State_New York"
]

X = df_enc[features].astype(float)
y = df_enc["Profit"].astype(float)

# Standard train-test split: 80% train, 20% test (random_state=0 for benchmark consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=0
)

# Standardized train/test for regularization rankings
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)


# ============================================================
# PHASE 3: 6 FEATURE SELECTION ALGORITHMS DEFINITION
# ============================================================
algorithms = {}

# 1. Lasso (L1 Regularization Ranking)
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=100000).fit(X_train_scaled, y_train)
lasso_rank = [f for _, f in sorted(zip(abs(lasso_cv.coef_), features), reverse=True)]
algorithms["Lasso"] = [lasso_rank[:k] for k in range(1, 6)]

# 2. Forward Stepwise Selection (SFS)
sfs_subsets = []
for k in range(1, 6):
    if k == 5:
        sfs_subsets.append(features)
    else:
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward")
        sfs.fit(X_train, y_train)
        sfs_subsets.append(list(X_train.columns[sfs.get_support()]))
algorithms["Forward SFS"] = sfs_subsets

# 3. Recursive Feature Elimination (RFE)
rfe_subsets = []
for k in range(1, 6):
    if k == 5:
        rfe_subsets.append(features)
    else:
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train_scaled, y_train)
        rfe_subsets.append(list(X_train.columns[rfe.get_support()]))
algorithms["RFE"] = rfe_subsets

# 4. ElasticNet (L1 + L2 Ranking)
enet_cv = ElasticNetCV(cv=5, random_state=0, max_iter=100000).fit(X_train_scaled, y_train)
enet_rank = [f for _, f in sorted(zip(abs(enet_cv.coef_), features), reverse=True)]
algorithms["ElasticNet"] = [enet_rank[:k] for k in range(1, 6)]

# 5. Ridge (L2 Shrinkage Ranking) - Added algorithm
ridge_cv = RidgeCV(cv=5).fit(X_train_scaled, y_train)
ridge_rank = [f for _, f in sorted(zip(abs(ridge_cv.coef_), features), reverse=True)]
algorithms["Ridge"] = [ridge_rank[:k] for k in range(1, 6)]

# 6. Exhaustive Best-Subset Selection
def evaluate_subset_rmse(sub):
    lr = LinearRegression().fit(X_train[list(sub)], y_train)
    preds = lr.predict(X_test[list(sub)])
    return np.sqrt(mean_squared_error(y_test, preds))

exhaustive_subsets = []
for k in range(1, 6):
    best_rmse = float("inf")
    best_sub = None
    for sub in itertools.combinations(features, k):
        rmse = evaluate_subset_rmse(sub)
        if rmse < best_rmse:
            best_rmse = rmse
            best_sub = list(sub)
    exhaustive_subsets.append(best_sub)
algorithms["Exhaustive"] = exhaustive_subsets

print("Feature progression subsets generated successfully across 6 algorithms.")


# ============================================================
# PHASE 4: DETAILED STATISTICAL INFERENCE & MODEL METRICS
# ============================================================
model_registry = {}

for algo_name, subsets in algorithms.items():
    model_registry[algo_name] = []
    
    for k_idx, subset in enumerate(subsets):
        k = k_idx + 1
        sub = list(subset)
        
        # Scikit-learn model for predictions & test metrics
        sk_model = LinearRegression()
        sk_model.fit(X_train[sub], y_train)
        y_test_pred = sk_model.predict(X_test[sub])
        
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))
        
        n_test = len(y_test)
        p_val = len(sub)
        test_adj_r2 = float(1 - (1 - test_r2) * (n_test - 1) / (n_test - p_val - 1)) if n_test > p_val + 1 else test_r2
        
        # Statsmodels OLS model on training set for inferential statistics
        X_train_sm = sm.add_constant(X_train[sub])
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        f_stat = float(sm_model.fvalue) if not np.isnan(sm_model.fvalue) else 0.0
        f_pvalue = float(sm_model.f_pvalue) if not np.isnan(sm_model.f_pvalue) else 1.0
        aic = float(sm_model.aic)
        bic = float(sm_model.bic)
        
        # Coefficients table with SE, t, p, CI, and VIF
        coef_list = []
        
        # Intercept
        intercept_ci = sm_model.conf_int().loc["const"]
        coef_list.append({
            "feature": "Intercept (β₀)",
            "coef": float(sm_model.params["const"]),
            "stderr": float(sm_model.bse["const"]),
            "tstat": float(sm_model.tvalues["const"]),
            "pvalue": float(sm_model.pvalues["const"]),
            "ci_lower": float(intercept_ci[0]),
            "ci_upper": float(intercept_ci[1]),
            "vif": None
        })
        
        # Features VIF calculation (on X_train[sub])
        vif_dict = {}
        if len(sub) > 1:
            X_vif_df = X_train[sub].copy()
            for i, col in enumerate(sub):
                try:
                    v = variance_inflation_factor(X_vif_df.values, i)
                    vif_dict[col] = float(v) if not np.isinf(v) and not np.isnan(v) else 999.0
                except Exception:
                    vif_dict[col] = 1.0
        else:
            vif_dict[sub[0]] = 1.0
            
        for feat in sub:
            ci = sm_model.conf_int().loc[feat]
            coef_list.append({
                "feature": feat,
                "coef": float(sm_model.params[feat]),
                "stderr": float(sm_model.bse[feat]),
                "tstat": float(sm_model.tvalues[feat]),
                "pvalue": float(sm_model.pvalues[feat]),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "vif": vif_dict.get(feat, 1.0)
            })
            
        # Residuals & Outliers analysis on Test set
        outliers = []
        for orig_idx, actual, pred in zip(X_test.index, y_test, y_test_pred):
            res = float(actual - pred)
            abs_err = abs(res)
            pct_err = float(abs_err / actual * 100) if actual != 0 else 0.0
            outliers.append({
                "index": int(orig_idx),
                "actual": float(actual),
                "predicted": float(pred),
                "residual": res,
                "abs_error": abs_err,
                "pct_error": pct_err
            })
            
        outliers.sort(key=lambda x: x["abs_error"], reverse=True)
        for r_i, o in enumerate(outliers):
            o["rank"] = r_i + 1
            
        model_registry[algo_name].append({
            "k": k,
            "features": sub,
            "rmse": test_rmse,
            "mae": test_mae,
            "mse": test_mse,
            "r2": test_r2,
            "adj_r2": test_adj_r2,
            "aic": aic,
            "bic": bic,
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "coefficients": coef_list,
            "outliers": outliers
        })

# Save JSON and JS assets
with open("model_data.json", "w", encoding="utf-8") as f:
    json.dump(model_registry, f, indent=2)

os.makedirs("static", exist_ok=True)
with open("static/model_data.js", "w", encoding="utf-8") as f:
    f.write("/* 50 Startups: Model Registry Data Asset */\n")
    f.write("window.MODEL_DATA = " + json.dumps(model_registry, indent=2) + ";\n")

print("Saved model_data.json and static/model_data.js successfully.")


# ============================================================
# PHASE 5: PUBLICATION-READY VISUALIZATIONS GENERATION
# ============================================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors = {
    "Lasso": "#4f46e5",
    "Forward SFS": "#06b6d4",
    "RFE": "#ec4899",
    "ElasticNet": "#8b5cf6",
    "Ridge": "#3b82f6",
    "Exhaustive": "#10b981"
}

# 1. All Algorithms RMSE & R2 Progression Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

for algo, records in model_registry.items():
    ks = [r["k"] for r in records]
    rmses = [r["rmse"] for r in records]
    r2s = [r["r2"] for r in records]
    
    ax1.plot(ks, rmses, marker='o', linewidth=2.2, label=algo, color=colors.get(algo, "#64748b"))
    ax2.plot(ks, r2s, marker='s', linewidth=2.2, label=algo, color=colors.get(algo, "#64748b"))

# Annotate Sweet Spot at k=2
ax1.axvline(x=2, color='#10b981', linestyle='--', alpha=0.7, label='Optimal k=2 Sweet Spot')
ax1.set_title("Test RMSE vs. Number of Features (k)", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel("Number of Features (k)", fontsize=11)
ax1.set_ylabel("Test Root Mean Squared Error ($)", fontsize=11)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.legend(loc="upper right", frameon=True, fontsize=9)

ax2.axvline(x=2, color='#10b981', linestyle='--', alpha=0.7, label='Optimal k=2 Sweet Spot')
ax2.set_title("Test R² vs. Number of Features (k)", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Number of Features (k)", fontsize=11)
ax2.set_ylabel("Test R-squared", fontsize=11)
ax2.set_xticks([1, 2, 3, 4, 5])
ax2.legend(loc="lower right", frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig("all_algorithms_comparison.png")
plt.close()

# 2. Actual vs. Predicted (k=2 Optimal Subset)
best_model = model_registry["Lasso"][1] # k=2
best_preds = [o["predicted"] for o in best_model["outliers"]]
best_actuals = [o["actual"] for o in best_model["outliers"]]

fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
ax.scatter(best_actuals, best_preds, color="#4f46e5", s=70, edgecolors="white", linewidth=1.5, zorder=5, label="Test Samples")
min_val = min(min(best_actuals), min(best_preds)) * 0.95
max_val = max(max(best_actuals), max(best_preds)) * 1.05
ax.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2, label="Identity Line (y = ŷ)")
ax.set_title("Actual vs. Predicted Profit (Optimal k=2 Model)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Actual Profit ($)", fontsize=11)
ax.set_ylabel("Predicted Profit ($)", fontsize=11)
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Visualizations generated: all_algorithms_comparison.png, actual_vs_predicted.png.")
print("=== DIC2 Pipeline Execution Finished Successfully! ===")
