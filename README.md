# 🌾 CA — Agro Agriculture Price Predictor

ML-powered web application for predicting agro modal prices across Indian agricultural markets.

---

## 📁 Project Structure

```
ca_ml_project/
├── app.py                  # Flask web server
├── train_model.py          # ML training pipeline
├── requirements.txt        # Python dependencies
├── data/
│   └── agro_prices.csv
├── models/                 # Saved models & metadata (auto-generated)
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_cols.pkl
│   ├── metrics.json
│   ├── categories.json
│   ├── agro_stats.json
│   └── state_stats.json
├── static/                 # Charts (auto-generated)
│   ├── chart_model_comparison.png
│   ├── chart_actual_vs_predicted.png
│   ├── chart_price_distribution.png
│   ├── chart_feature_importance.png
│   ├── chart_top_agros.png
│   └── chart_state_prices.png
└── templates/
    └── index.html          # Dashboard UI
```

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the model
```bash
python train_model.py
```

### Step 3 — Start the web server
```bash
python app.py
```

### Step 4 — Open in browser
```
http://localhost:5000
```

---

## 🤖 ML Models Trained

| Model               | MAE (₹) | RMSE (₹) | R² Score |
|---------------------|---------|----------|----------|
| Linear Regression   | 119.15  | 249.96   | 0.9967   |
| Ridge Regression    | 119.15  | 249.96   | 0.9967   |
| **Random Forest**   | **85.87** | **215.42** | **0.9976** ✅ |
| Gradient Boosting   | 108.91  | 234.93   | 0.9971   |

**Best Model: Random Forest** (automatically deployed)

---

## 🎯 Features

- Predict modal price for any agro, state, district, market combination
- Cascading dropdowns (State → District → Market)
- Live prediction history chart
- 6 analysis charts (model comparison, actual vs predicted, distributions, feature importance)
- Model performance comparison table
- Agro & state price statistics

---

## 📊 Dataset

- **5,947 rows** | **10 columns**
- **205 unique agros** (Wheat, Tomato, Onion, Potato…)
- **26 Indian states**
- Prices: Min, Max, Modal (₹/quintal)

---

## 🚀 Deployment (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

For cloud deployment (Render, Railway, Heroku):
- Add a `Procfile`: `web: gunicorn app:app`
- Set `PORT` environment variable
