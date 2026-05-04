"""
register_model.py — Register best model to MLflow Model Registry.
Saves version + run_id to results/step3_registration.json.

Run AFTER train.py (or tune.py).
"""

import json, os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH        = "models/best_model.pkl"
RESULTS_PATH      = "results/step4_s6.json"
REGISTERED_NAME   = "quantumbench-circuit-exec-ms-predictor"
EXPERIMENT        = "quantumbench-circuit-exec-ms"
# ─────────────────────────────────────────────────────────────────────────────


def get_best_run_id():
    """Fetch the run with the lowest RMSE from the experiment."""
    client = MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT}' not found. Run train.py first."
        )

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.RMSE ASC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found. Run train.py first.")

    best = runs[0]
    print(f"[INFO] Best run_id={best.info.run_id}  "
          f"RMSE={best.data.metrics.get('RMSE', 'N/A')}")
    return best.info.run_id


def register():
    os.makedirs("results", exist_ok=True)

    run_id     = get_best_run_id()
    model_uri  = f"runs:/{run_id}/model"
    client     = MlflowClient()

    # ── Log the model artifact under the best run (if not already logged) ─────
    with mlflow.start_run(run_id=run_id):
        model = joblib.load(MODEL_PATH)
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=REGISTERED_NAME)

    # ── Fetch the version that was just created ───────────────────────────────
    versions = client.get_latest_versions(REGISTERED_NAME)
    version  = versions[-1].version if versions else "1"

    print(f"[INFO] Registered '{REGISTERED_NAME}'  version={version}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    output = {
    "registered_model_name": REGISTERED_NAME,
    "version":               int(version),
    "run_id":                run_id,
    "source_metric":         "mae",
    "source_metric_value":   round(float(versions[-1].tags.get("mae", 0.0)), 4),
}
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Registration results → {RESULTS_PATH}")


if __name__ == "__main__":
    register()
