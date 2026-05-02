"""
CA Agro Price Prediction - Flask Web Application
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

#  Load Artefacts 
model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
feature_cols  = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json"))        as f: metrics   = json.load(f)
with open(os.path.join(MODEL_DIR, "categories.json"))     as f: categories = json.load(f)
with open(os.path.join(MODEL_DIR, "agro_stats.json")) as f: agro_stats = json.load(f)
with open(os.path.join(MODEL_DIR, "state_stats.json"))    as f: state_stats = json.load(f)

#  Routes 

@app.route("/")
def index():
    return render_template("index.html",
                           categories=categories,
                           metrics=metrics,
                           agro_stats=agro_stats,
                           state_stats=state_stats)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        cat_cols = ['State', 'District', 'Market', 'Agro', 'Variety', 'Grade']

        row = {}
        for col in cat_cols:
            val = data.get(col, "")
            le  = label_encoders[col]
            if val in le.classes_:
                row[col + "_enc"] = int(le.transform([val])[0])
            else:
                row[col + "_enc"] = 0

        row["Min_Price"] = float(data.get("Min_Price", 0))
        row["Max_Price"] = float(data.get("Max_Price", 0))

        X = np.array([[row[f] for f in feature_cols]])
        prediction = float(model.predict(X)[0])

        return jsonify({
            "success": True,
            "modal_price": round(prediction, 2),
            "min_price": row["Min_Price"],
            "max_price": row["Max_Price"],
            "agro": data.get("Agro", ""),
            "state": data.get("State", ""),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(metrics)


@app.route("/api/comm_stats")
def api_agro_stats():
    return jsonify(agro_stats)


@app.route("/api/state_stats")
def api_state_stats():
    return jsonify(state_stats)


@app.route("/stats")
def stats_page():
    stat_path = os.path.join(MODEL_DIR, "stat_tests.json")
    if not os.path.exists(stat_path):
        return "Statistical tests not yet run. Please run: python statistical_tests.py", 404
    with open(stat_path) as f:
        stat_tests = json.load(f)
    return render_template("stats.html", stat_tests=stat_tests)

@app.route("/api/stat_tests")
def api_stat_tests():
    stat_path = os.path.join(MODEL_DIR, "stat_tests.json")
    if not os.path.exists(stat_path):
        return jsonify({"error": "Run statistical_tests.py first"}), 404
    with open(stat_path) as f:
        return jsonify(json.load(f))

@app.route("/api/districts")
def api_districts():
    state = request.args.get("state", "")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "agro_prices.csv"))
    df.columns = df.columns.str.replace('_x0020_', '_')
    if state:
        districts = sorted(df[df["State"] == state]["District"].unique().tolist())
    else:
        districts = sorted(df["District"].unique().tolist())
    return jsonify(districts)


@app.route("/api/markets")
def api_markets():
    district = request.args.get("district", "")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "agro_prices.csv"))
    df.columns = df.columns.str.replace('_x0020_', '_')
    if district:
        markets = sorted(df[df["District"] == district]["Market"].unique().tolist())
    else:
        markets = sorted(df["Market"].unique().tolist())
    return jsonify(markets)


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  [SERVER] CA Agro Price Prediction Server")
    print("   Visit: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
