"""
Statistical Analysis  Agro Price Prediction
=========================================================
Focus: Z-Test, T-Test, and Linear Regression
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "agro_prices.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC    = os.path.join(BASE_DIR, "static")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC,    exist_ok=True)

print("=" * 62)
print("  Statistical Analysis  Agro Price Dataset")
print("=" * 62)

#  LOAD DATA 
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.replace('_x0020_', '_')
df.dropna(inplace=True)

modal     = df['Modal_Price']
MU0       = 2000        # benchmark hypothesis mean
ALPHA     = 0.05
N         = len(modal)

results = {}   # collected for JSON export

# 
# TEST 1  One-Sample Z-Test
# 
print("\n[Test 1] One-Sample Z-Test (target = 2,000) ")
xbar = modal.mean()
std  = modal.std()
se   = std / np.sqrt(N)
z    = (xbar - MU0) / se
z_p  = 2 * (1 - norm.cdf(abs(z)))
z_p  = max(z_p, 1e-300)
z_decision = "Reject H0" if z_p < ALPHA else "Fail to Reject H0"
results['z_test'] = {
    "name":         "One-Sample Z-Test",
    "h0":           "Mean Modal_Price = 2,000",
    "h1":           "Mean Modal_Price != 2,000",
    "formula":      "Z = (x - mu) / (sigma / sqrt(n))",
    "sample_mean":  round(float(xbar), 2),
    "mu0":          MU0,
    "std":          round(float(std), 2),
    "n":            N,
    "statistic":    round(float(z), 4),
    "p_value":      round(float(z_p), 6),
    "alpha":        ALPHA,
    "decision":     z_decision,
    "interpretation": f"Sample mean is {xbar:,.2f}. The Z-statistic of {z:.2f} indicates a significant difference from the benchmark."
}
print(f"   Z = {z:.4f},  p = {z_p:.6f}  ->  {z_decision}")

# 
# TEST 2  One-Sample T-Test
# 
print("\n[Test 2] One-Sample T-Test (target = 2,000) ")
t1, t1_p = stats.ttest_1samp(modal, MU0)
t1_decision = "Reject H0" if t1_p < ALPHA else "Fail to Reject H0"
results['t_test'] = {
    "name":         "One-Sample T-Test",
    "h0":           "Mean Modal_Price = 2,000",
    "h1":           "Mean Modal_Price != 2,000",
    "formula":      "t = (x - mu) / (s / sqrt(n))",
    "sample_mean":  round(float(xbar), 2),
    "mu0":          MU0,
    "df":           N - 1,
    "statistic":    round(float(t1), 4),
    "p_value":      round(float(t1_p), 6),
    "alpha":        ALPHA,
    "decision":     t1_decision,
    "interpretation": f"t = {t1:.2f}. The T-test confirms the Z-test results for this large sample."
}
print(f"   t = {t1:.4f},  p = {t1_p:.6f}  ->  {t1_decision}")

# 
# ANALYSIS 3  Linear Regression
# 
print("\n[Analysis 3] Linear Regression (Min_Price vs Modal_Price) ")
X = df[['Min_Price']].values
y = df['Modal_Price'].values

model_lr = LinearRegression()
model_lr.fit(X, y)
y_pred = model_lr.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
slope = model_lr.coef_[0]
intercept = model_lr.intercept_

results['linear_regression'] = {
    "name": "Linear Regression Analysis",
    "independent_var": "Min_Price",
    "dependent_var": "Modal_Price",
    "formula": f"y = {slope:.4f}x + {intercept:.2f}",
    "r2_score": round(float(r2), 4),
    "mae": round(float(mae), 2),
    "slope": round(float(slope), 4),
    "intercept": round(float(intercept), 2),
    "interpretation": f"R-squared of {r2:.4f} shows that Min_Price explains {r2*100:.1f}% of the variance in Modal_Price."
}
print(f"   R2 = {r2:.4f},  MAE = {mae:.2f}")

#  SAVE JSON 
results['_meta'] = {
    "dataset_n": N,
    "alpha":     ALPHA,
    "mu0":       MU0,
    "dataset_mean": round(float(xbar), 2),
    "dataset_std":  round(float(std), 2),
    "dataset_median": round(float(modal.median()), 2),
}
with open(os.path.join(MODEL_DIR, "stat_tests.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[SUCCESS] Saved results -> models/stat_tests.json")

#  CHARTS 
print("\n[Charts] Generating simplified analysis charts ")

# --- CHARTS STYLING ---
plt.style.use('dark_background')
BG_COLOR = '#1F1510'  # Match the deeper dark background
TEXT_COLOR = '#F8F1E5'

matplotlib.rcParams['axes.facecolor'] = BG_COLOR
matplotlib.rcParams['figure.facecolor'] = BG_COLOR
matplotlib.rcParams['text.color'] = TEXT_COLOR
matplotlib.rcParams['axes.labelcolor'] = TEXT_COLOR
matplotlib.rcParams['xtick.color'] = TEXT_COLOR
matplotlib.rcParams['ytick.color'] = TEXT_COLOR
matplotlib.rcParams['grid.color'] = '#32251E'

# Chart 1: Z-Test Distribution
fig, ax = plt.subplots(figsize=(10, 5))
x_vals = np.linspace(-5, 5, 400)
ax.plot(x_vals, norm.pdf(x_vals), color='#2C3340', lw=2, label='Standard Normal N(0,1)')
ax.fill_between(x_vals, norm.pdf(x_vals), where=(x_vals < -1.96), color='#B01020', alpha=0.35, label='Rejection region')
ax.fill_between(x_vals, norm.pdf(x_vals), where=(x_vals >  1.96), color='#B01020', alpha=0.35)
z_clipped = max(min(z, 4.8), -4.8)
ax.axvline(x=z_clipped, color='#2E75B6', lw=2.5, linestyle='--', label=f'Z = {z:.2f}')
ax.set_title('One-Sample Z-Test Analysis', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC, 'stat_ztest.png'), dpi=130)
plt.close()

# Chart 2: T-Test Distribution
fig, ax = plt.subplots(figsize=(10, 5))
x_vals = np.linspace(-5, 5, 400)
# Use t-distribution with N-1 degrees of freedom
ax.plot(x_vals, stats.t.pdf(x_vals, df=N-1), color='#2C3340', lw=2, label=f'T-Dist (df={N-1})')
# Critical values for t-dist are very close to 1.96 for large N
t_crit = stats.t.ppf(0.975, df=N-1)
ax.fill_between(x_vals, stats.t.pdf(x_vals, df=N-1), where=(x_vals < -t_crit), color='#B01020', alpha=0.35, label='Rejection region')
ax.fill_between(x_vals, stats.t.pdf(x_vals, df=N-1), where=(x_vals > t_crit), color='#B01020', alpha=0.35)
t_clipped = max(min(t1, 4.8), -4.8)
ax.axvline(x=t_clipped, color='#E8810A', lw=2.5, linestyle='--', label=f't = {t1:.2f}')
ax.set_title('One-Sample T-Test Analysis', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC, 'stat_ttest.png'), dpi=130)
plt.close()

# Chart 3: Linear Regression Plot
fig, ax = plt.subplots(figsize=(10, 6))
sample_idx = np.random.choice(len(df), min(1000, len(df)), replace=False)
ax.scatter(df['Min_Price'].iloc[sample_idx], df['Modal_Price'].iloc[sample_idx], alpha=0.3, color='#B01020', s=10, label='Data points')
x_range = np.linspace(df['Min_Price'].min(), df['Min_Price'].max(), 100).reshape(-1, 1)
y_range = model_lr.predict(x_range)
ax.plot(x_range, y_range, color='#2C3340', lw=3, label='Regression Line')
ax.set_title(f'Linear Regression: R2 = {r2:.4f}', fontsize=12, fontweight='bold')
ax.set_xlabel('Min Price (Input)')
ax.set_ylabel('Modal Price (Target)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC, 'stat_regression.png'), dpi=130)
plt.close()

print("   Saved: stat_ztest.png, stat_ttest.png, stat_regression.png")
print("\n" + "=" * 62)
print("  [SUCCESS] Analysis Complete!")
print("=" * 62)
