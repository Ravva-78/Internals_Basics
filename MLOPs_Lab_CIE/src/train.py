"""
train.py — Train Lasso & GradientBoostingRegressor, log to MLflow, save best model.
"""

import json, os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/training_data.csv"
MODEL_PATH   = "models/best_model.pkl"
RESULTS_PATH = "results/step1_s1.json"
TARGET       = "circuit_exec_ms"
EXPERIMENT   = "quantumbench-circuit-exec-ms"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
# ─────────────────────────────────────────────────────────────────────────────


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_training():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[INFO] Train={len(X_train)}  Test={len(X_test)}")

    mlflow.set_experiment(EXPERIMENT)

    candidates = {
        "Lasso": Lasso(alpha=1.0, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE
        ),
    }

    results    = []
    best_name  = None
    best_mae   = float("inf")
    best_model = None

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("project_phase", "model_selection")
            mlflow.log_params(
                {k: v for k, v in model.get_params().items() if v is not None}
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            rms = compute_rmse(y_test, y_pred)
            mlflow.log_metric("MAE",  mae)
            mlflow.log_metric("RMSE", rms)

            run_id = mlflow.active_run().info.run_id
            print(f"[{name}]  MAE={mae:.4f}  RMSE={rms:.4f}  run_id={run_id}")
            results.append({"model": name, "MAE": round(mae, 4),
                            "RMSE": round(rms, 4), "run_id": run_id})

            if mae < best_mae:
                best_mae   = mae
                best_name  = name
                best_model = model

    # ── Persist best model ────────────────────────────────────────────────────
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[BEST] {best_name}  MAE={best_mae:.4f}  → {MODEL_PATH}")

    output = {
        "experiment_name": EXPERIMENT,
        "models": [
            {"name": r["model"], "mae": r["MAE"], "rmse": r["RMSE"]}
            for r in results
        ],
        "best_model":        best_name,
        "best_metric_name":  "mae",
        "best_metric_value": round(best_mae, 4),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run_training()
