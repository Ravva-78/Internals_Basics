"""
predict_cli.py — CLI tool to predict circuit execution time from quantum features.
Usage:
    python src/predict_cli.py --qubit_count 8 --gate_depth 25 \
                               --error_rate_pct 2.5 --is_error_corrected 0
"""

import argparse
import sys
import numpy as np
import joblib

MODEL_PATH = "models/best_model.pkl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict quantum circuit execution time (ms)."
    )
    parser.add_argument("--qubit_count",        type=int,   required=True,
                        help="Number of qubits (e.g. 8)")
    parser.add_argument("--gate_depth",         type=int,   required=True,
                        help="Circuit gate depth (e.g. 25)")
    parser.add_argument("--error_rate_pct",     type=float, required=True,
                        help="Error rate percentage (e.g. 2.5)")
    parser.add_argument("--is_error_corrected", type=int,   required=True,
                        choices=[0, 1],
                        help="Error correction applied? 1=yes 0=no")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at '{MODEL_PATH}'. Run train.py first.")
        sys.exit(1)

    features = np.array([[
        args.qubit_count,
        args.gate_depth,
        args.error_rate_pct,
        args.is_error_corrected,
    ]])

    prediction = model.predict(features)[0]
    print(f"\n  Predicted Execution Time: {prediction:.2f} ms\n")


if __name__ == "__main__":
    main()
