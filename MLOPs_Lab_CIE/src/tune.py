"""
tune.py — Hyperparameter tuning for GradientBoostingRegressor via RandomizedSearchCV.
Uses MLflow nested runs. Saves best params + MAE to results/.
"""

import json, os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/training_data.csv"
MODEL_PATH   = "models/best_model.pkl"
RESULTS_PATH = "results/step2_s2.json"          # ← was step2_tuning.json
TARGET       = "circuit_exec_ms"                 # ← was exec_time_ms
EXPERIMENT   = "quantumbench-circuit-exec-ms"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_ITER       = 10
CV_FOLDS     = 5

PARAM_GRID = {
    "n_estimators":  [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [3, 5, 10],
}
# ─────────────────────────────────────────────────────────────────────────────


def neg_mae_scorer():                            # ← was neg_rmse_scorer
    def _mae(y_true, y_pred):
        return -float(mean_absolute_error(y_true, y_pred))
    return make_scorer(_mae)


def run_tuning():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="tuning-quantumbench"):  # ← was RandomizedSearch_GBR
        mlflow.set_tag("project_phase", "hyperparameter_tuning")
        mlflow.log_param("cv_folds",   CV_FOLDS)
        mlflow.log_param("n_iter",     N_ITER)
        mlflow.log_param("param_grid", str(PARAM_GRID))

        base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        search     = RandomizedSearchCV(
            estimator           = base_model,
            param_distributions = PARAM_GRID,
            n_iter              = N_ITER,
            cv                  = CV_FOLDS,
            scoring             = neg_mae_scorer(),          # ← MAE scorer
            n_jobs              = -1,
            random_state        = RANDOM_STATE,
            refit               = True,
        )
        search.fit(X_train, y_train)

        # ── Log each CV candidate as a nested run ─────────────────────────────
        for i, (params, score) in enumerate(
            zip(search.cv_results_["params"],
                search.cv_results_["mean_test_score"])
        ):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_mean_mae", round(-score, 4))  # ← was cv_mean_rmse

        # ── Evaluate best estimator on held-out test set ──────────────────────
        best        = search.best_estimator_
        y_pred      = best.predict(X_test)
        test_mae    = float(mean_absolute_error(y_test, y_pred))   # ← was RMSE
        best_params = search.best_params_
        cv_mae      = round(-search.best_score_, 4)                # ← was cv_rmse

        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_mae", cv_mae)                   # ← was best_cv_rmse
        mlflow.log_metric("test_mae",    round(test_mae, 4))       # ← was test_rmse

        print(f"[BEST PARAMS] {best_params}")
        print(f"[CV MAE]      {cv_mae}")
        print(f"[TEST MAE]    {test_mae:.4f}")

        joblib.dump(best, MODEL_PATH)
        print(f"[INFO] Tuned model saved → {MODEL_PATH}")

        # ── Save results JSON (exact format from question paper) ──────────────
        output = {
            "search_type":      "random",
            "n_folds":          CV_FOLDS,
            "total_trials":     N_ITER,
            "best_params":      best_params,
            "best_mae":         round(test_mae, 4),
            "best_cv_mae":      cv_mae,
            "parent_run_name":  "tuning-quantumbench",
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[INFO] Tuning results → {RESULTS_PATH}")


if __name__ == "__main__":
    run_tuning()