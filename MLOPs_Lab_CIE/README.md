# MLOps Lab CIE — QuantumBench Circuit Execution Time Predictor

**USN:** 1BM24AI414  
**Course:** MLOps (24AM6AEMLO)  
**Date:** 04th May 2026

## Problem Statement
Predict quantum circuit execution time (ms) based on circuit features.

## Dataset Features
- `qubit_count` — (5–100)
- `gate_depth` — (10–500)
- `error_rate_pct` — (0.1–5)
- `is_error_corrected` — (0–1)
- **Target:** `circuit_exec_ms`

## Tasks
| Task | Description | Result |
|------|-------------|--------|
| 1 | Experiment Tracking & Model Comparison | `results/step1_s1.json` |
| 2 | Hyperparameter Tuning | `results/step2_s2.json` |
| 3 | Docker Packaging | `results/step3_s3.json` |
| 4 | Model Versioning | `results/step4_s6.json` |

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Task 1 — Train models
```bash
python src/train.py
```

### Task 2 — Hyperparameter tuning
```bash
python src/tune.py
```

### Task 3 — Docker
```bash
docker build -t quantumbench-predictor:v1 .
docker run quantumbench-predictor:v1 --qubit_count 71 --gate_depth 350 --error_rate_pct 2.4 --is_error_corrected 0
```

### Task 4 — Register model
```bash
python src/register_model.py
```