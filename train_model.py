"""
CA (Agro Agriculture) Price Prediction - ML Training Script
Dataset: Indian Agricultural Market Prices
Target: Modal Price Prediction
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 
# PATHS
# 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "agro_prices.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

print("=" * 60)
print("  CA - Agro Agriculture Price Prediction ML Pipeline")
print("=" * 60)

# 
# 1. LOAD & EXPLORE DATA
# 
print("\n[1] Loading Dataset...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace('_x0020_', '_')  # Clean column names
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Unique Agros: {df['Agro'].nunique()}")
print(f"    Unique States: {df['State'].nunique()}")

# 
# 2. PREPROCESSING
# 
print("\n[2] Preprocessing...")
df.dropna(inplace=True)

# Encode categorical columns
cat_cols = ['State', 'District', 'Market', 'Agro', 'Variety', 'Grade']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and Target
feature_cols = [c + '_enc' for c in cat_cols] + ['Min_Price', 'Max_Price']
target_col = 'Modal_Price'

X = df[feature_cols]
y = df[target_col]

print(f"    Features: {feature_cols}")
print(f"    Target: {target_col}")
print(f"    Samples: {len(X)}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

# 
# 3. TRAIN MULTIPLE MODELS
# 
print("\n[3] Training Models...")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"    Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[name] = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "model": model,
        "y_pred": y_pred,
    }
    print(f"      MAE={mae:.2f}  RMSE={rmse:.2f}  R={r2:.4f}")

# 
# 4. BEST MODEL
# 
best_name = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
print(f"\n[4] Best Model: {best_name}  (R={results[best_name]['R2']})")

# Save best model + encoders + metadata
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))

# Save metrics as JSON for dashboard
metrics_out = {
    k: {"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"]}
    for k, v in results.items()
}
metrics_out["best_model"] = best_name
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_out, f)

# Save category lists for UI dropdowns
cat_options = {col: sorted(df[col].unique().tolist()) for col in cat_cols}
with open(os.path.join(MODEL_DIR, "categories.json"), "w") as f:
    json.dump(cat_options, f)

# Save agro stats for charts
agro_stats = (
    df.groupby("Agro")[["Min_Price", "Max_Price", "Modal_Price"]]
    .mean()
    .round(2)
    .reset_index()
    .sort_values("Modal_Price", ascending=False)
    .head(20)
)
agro_stats.to_json(os.path.join(MODEL_DIR, "agro_stats.json"), orient="records")

state_stats = (
    df.groupby("State")[["Modal_Price"]]
    .mean()
    .round(2)
    .reset_index()
    .sort_values("Modal_Price", ascending=False)
)
state_stats.to_json(os.path.join(MODEL_DIR, "state_stats.json"), orient="records")

# 
# 5. GENERATE CHARTS
# 
# --- CHARTS STYLING ---
plt.style.use('dark_background')
BG_COLOR = '#1F1510'  # Match the deeper dark background
TEXT_COLOR = '#F8F1E5'
ACCENT_COLOR = '#D4AF37'

matplotlib.rcParams['axes.facecolor'] = BG_COLOR
matplotlib.rcParams['figure.facecolor'] = BG_COLOR
matplotlib.rcParams['text.color'] = TEXT_COLOR
matplotlib.rcParams['axes.labelcolor'] = TEXT_COLOR
matplotlib.rcParams['xtick.color'] = TEXT_COLOR
matplotlib.rcParams['ytick.color'] = TEXT_COLOR
matplotlib.rcParams['grid.color'] = '#32251E'

colors = ['#D4AF37', '#E07A5F', '#84A98C', '#CAD2C5', '#8338ec']

# --- Chart 1: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7)) # Increased size
model_names = list(metrics_out.keys() - {"best_model"})
maes  = [metrics_out[m]["MAE"]  for m in model_names]
rmses = [metrics_out[m]["RMSE"] for m in model_names]
r2s   = [metrics_out[m]["R2"]   for m in model_names]

axes[0].bar(model_names, maes, color=colors[:len(model_names)])
axes[0].set_title("MAE Comparison (lower = better)", fontweight='bold', fontsize=14)
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(model_names, rmses, color=colors[:len(model_names)])
axes[1].set_title("RMSE Comparison (lower = better)", fontweight='bold', fontsize=14)
axes[1].set_ylabel("RMSE")
axes[1].tick_params(axis='x', rotation=30)

axes[2].bar(model_names, r2s, color=colors[:len(model_names)])
axes[2].set_title("R2 Score (higher = better)", fontweight='bold', fontsize=14)
axes[2].set_ylabel("R2 Score")
axes[2].set_ylim(0, 1.1)
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_model_comparison.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_model_comparison.png")

# --- Chart 2: Correlation Heatmap ---
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=0.5)
plt.title("Dataset Correlation Heatmap", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_correlation_heatmap.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_correlation_heatmap.png")

# --- Chart 3: Actual vs Predicted ---
best_pred = results[best_name]["y_pred"]
sample_idx = np.random.choice(len(y_test), min(200, len(y_test)), replace=False)
y_test_arr = np.array(y_test)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test_arr[sample_idx], best_pred[sample_idx], alpha=0.6, color='#4f86c6', edgecolors='white', s=60)
lims = [min(y_test_arr.min(), best_pred.min()), max(y_test_arr.max(), best_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel("Actual Modal Price", fontsize=12)
ax.set_ylabel("Predicted Modal Price", fontsize=12)
ax.set_title(f"Actual vs Predicted - {best_name}", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_actual_vs_predicted.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_actual_vs_predicted.png")

# --- Chart 4: Top 15 Agros by Modal Price ---
top15 = agro_stats.head(15)
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top15["Agro"], top15["Modal_Price"], color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15)))
ax.set_xlabel("Average Modal Price", fontsize=12)
ax.set_title("Top 15 Agros by Average Modal Price", fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, val in zip(bars, top15["Modal_Price"]):
    ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
            f'{val:,.0f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_top_agros.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_top_agros.png")

# --- Chart 5: Price Distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 7)) # Increased size
price_cols = ["Min_Price", "Max_Price", "Modal_Price"]
titles = ["Min Price Distribution", "Max Price Distribution", "Modal Price Distribution"]
for ax, col, title, color in zip(axes, price_cols, titles, ['#4f86c6', '#f4a261', '#2a9d8f']):
    clipped = df[col].clip(upper=df[col].quantile(0.95))
    ax.hist(clipped, bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_price_distribution.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_price_distribution.png")

# --- Chart 6: Feature Importance (Random Forest) ---
if "Random Forest" in results:
    rf_model = results["Random Forest"]["model"]
    importances = rf_model.feature_importances_
    # Clean feature names for display
    display_cols = [c.replace('_enc', '') for c in feature_cols]
    feat_df = pd.DataFrame({"Feature": display_cols, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color='#8338ec')
    ax.set_title("Feature Importance - Random Forest", fontsize=15, fontweight='bold')
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, "chart_feature_importance.png"), dpi=130, bbox_inches='tight')
    plt.close()
    print("    Saved: chart_feature_importance.png")

# --- Chart 7: State-wise Average Modal Price ---
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(state_stats["State"], state_stats["Modal_Price"],
              color=plt.cm.Blues(np.linspace(0.4, 0.9, len(state_stats))))
ax.set_title("State-wise Average Modal Price", fontsize=14, fontweight='bold')
ax.set_ylabel("Avg Modal Price")
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, state_stats["Modal_Price"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
            f'{val:,.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, "chart_state_prices.png"), dpi=130, bbox_inches='tight')
plt.close()
print("    Saved: chart_state_prices.png")

print("\n" + "=" * 60)
print("  [SUCCESS] Training Complete!")
print(f"  Best Model : {best_name}")
print(f"  R Score   : {results[best_name]['R2']}")
print(f"  MAE        : {results[best_name]['MAE']}")
print(f"  Models saved to: {MODEL_DIR}")
print(f"  Charts saved to: {STATIC_DIR}")
print("=" * 60)
