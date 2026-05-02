# Prediction Analysis

Run this script to generate evaluation charts from `predictions_log.json`.

## Usage

```bash
python3 scripts/analyze_predictions.py
python3 scripts/analyze_predictions.py --symbol SNDK
python3 scripts/analyze_predictions.py --symbol MRVL
```

## Outputs

Saved under `reports/`:
- `residual_vs_predicted*.png`
- `actual_vs_predicted*.png`
- `residual_histogram*.png`
- `summary*.json`
