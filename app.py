import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DATA_PATH = os.path.join(BASE_DIR, "model_data.json")
DATASET_PATH = os.path.join(BASE_DIR, "50_Startups.csv")

# Load model registry
try:
    with open(MODEL_DATA_PATH, "r", encoding="utf-8") as f:
        MODEL_DATA = json.load(f)
except Exception as e:
    print(f"Warning: Could not load model_data.json: {e}")
    MODEL_DATA = {}

# Load raw dataset
try:
    RAW_DATA = pd.read_csv(DATASET_PATH)
except Exception as e:
    print(f"Warning: Could not load 50_Startups.csv: {e}")
    RAW_DATA = pd.DataFrame()


@app.route("/")
def index():
    """Render the main interactive regression explorer dashboard."""
    return render_template("index.html", initial_algo="Lasso", initial_k=2)


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return the complete 6-algorithm feature selection evaluation data."""
    algo = request.args.get("algo")
    if algo and algo in MODEL_DATA:
        return jsonify({algo: MODEL_DATA[algo]})
    return jsonify(MODEL_DATA)


@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Return cross-algorithm comparison benchmark metrics."""
    summary = []
    for algo, records in MODEL_DATA.items():
        for r in records:
            summary.append({
                "algorithm": algo,
                "k": r["k"],
                "features": r["features"],
                "rmse": r["rmse"],
                "r2": r["r2"],
                "mae": r["mae"],
                "f_stat": r["f_stat"]
            })
    return jsonify(summary)


@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    """Return raw 50 Startups dataset records and summary stats."""
    if RAW_DATA.empty:
        return jsonify({"error": "Dataset unavailable"}), 404
        
    records = RAW_DATA.to_dict(orient="records")
    stats = {
        "total_records": len(RAW_DATA),
        "mean_profit": float(RAW_DATA["Profit"].mean()),
        "median_profit": float(RAW_DATA["Profit"].median()),
        "std_profit": float(RAW_DATA["Profit"].std()),
        "mean_rd": float(RAW_DATA["R&D Spend"].mean()),
        "mean_marketing": float(RAW_DATA["Marketing Spend"].mean()),
        "mean_admin": float(RAW_DATA["Administration"].mean())
    }
    return jsonify({"records": records, "stats": stats})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Live prediction endpoint: Calculates startup profit based on inputs,
    selected feature selection algorithm, and subset size k.
    """
    try:
        data = request.get_json(force=True) or {}
        algo = data.get("algo", "Lasso")
        k = int(data.get("k", 2))
        rd_spend = float(data.get("rd_spend", 100000))
        marketing_spend = float(data.get("marketing_spend", 150000))
        admin_spend = float(data.get("admin_spend", 120000))
        state = data.get("state", "California")

        if algo not in MODEL_DATA or k < 1 or k > len(MODEL_DATA[algo]):
            return jsonify({"error": "Invalid algorithm or k parameter"}), 400

        model_spec = MODEL_DATA[algo][k - 1]
        coefficients = model_spec["coefficients"]
        rmse = model_spec["rmse"]

        predicted_profit = 0.0
        breakdown = []

        for c in coefficients:
            feat_name = c["feature"]
            coef_val = c["coef"]

            if "Intercept" in feat_name:
                predicted_profit += coef_val
                breakdown.append({
                    "feature": "Intercept (β₀)",
                    "value": 1.0,
                    "coef": coef_val,
                    "contribution": coef_val
                })
            elif feat_name == "R&D Spend":
                contrib = coef_val * rd_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "R&D Spend",
                    "value": rd_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Marketing Spend":
                contrib = coef_val * marketing_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "Marketing Spend",
                    "value": marketing_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "Administration":
                contrib = coef_val * admin_spend
                predicted_profit += contrib
                breakdown.append({
                    "feature": "Administration",
                    "value": admin_spend,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "State_Florida":
                val = 1.0 if state == "Florida" else 0.0
                contrib = coef_val * val
                predicted_profit += contrib
                breakdown.append({
                    "feature": "State (Florida)",
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })
            elif feat_name == "State_New York":
                val = 1.0 if state == "New York" else 0.0
                contrib = coef_val * val
                predicted_profit += contrib
                breakdown.append({
                    "feature": "State (New York)",
                    "value": val,
                    "coef": coef_val,
                    "contribution": contrib
                })

        # Calculate ~95% prediction interval based on model RMSE (± 1.96 * RMSE)
        ci_half_width = 1.96 * rmse
        ci_lower = predicted_profit - ci_half_width
        ci_upper = predicted_profit + ci_half_width

        return jsonify({
            "algorithm": algo,
            "k": k,
            "active_features": model_spec["features"],
            "predicted_profit": predicted_profit,
            "prediction_interval_95": {
                "lower": ci_lower,
                "upper": ci_upper,
                "half_width": ci_half_width
            },
            "rmse": rmse,
            "r2": model_spec["r2"],
            "breakdown": breakdown
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting DIC2 Multiple Linear Regression Platform on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
