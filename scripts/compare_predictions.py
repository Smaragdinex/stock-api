from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from analysis_utils import DEFAULT_ACTUALS_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_PREDICTIONS_PATH, ensure_path, load_actuals, load_predictions


PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH
ACTUALS_PATH = DEFAULT_ACTUALS_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def main() -> int:
    ensure_path(PREDICTIONS_PATH, default_content="[]")
    ensure_path(ACTUALS_PATH, default_content="[]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_df = load_predictions(PREDICTIONS_PATH)
    actual_df = load_actuals(ACTUALS_PATH)

    if pred_df.empty or actual_df.empty:
        print(json.dumps({"error": "Missing prediction or actual data"}, ensure_ascii=False, indent=2))
        return 1

    pred_df = pred_df.sort_values(["symbol", "predicted_at"]).drop_duplicates(subset=["symbol"], keep="last")
    actual_df = actual_df.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol"], keep="last")

    merged = pd.merge(pred_df, actual_df, on="symbol", how="inner")
    if merged.empty:
        print(json.dumps({"error": "No overlapping symbols"}, ensure_ascii=False, indent=2))
        return 1

    merged["residual_close"] = merged["actualClose"] - merged["predictedMid"]
    rmse = mean_squared_error(merged["actualClose"], merged["predictedMid"]) ** 0.5
    r2 = r2_score(merged["actualClose"], merged["predictedMid"]) if len(merged) >= 2 else None

    summary = {
        "sampleSize": int(len(merged)),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6) if r2 is not None else None,
        "symbols": merged["symbol"].tolist(),
        "rows": merged[["symbol", "predictedMid", "actualClose", "residual_close"]].to_dict(orient="records"),
        "note": "Predictions have been corrected by actual market closes before comparison.",
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
