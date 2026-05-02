from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from analysis_utils import DEFAULT_PREDICTIONS_PATH, ensure_path, load_json_list


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "price_model.joblib"


def load_training_frame() -> pd.DataFrame:
    rows = []
    for item in load_json_list(PREDICTIONS_PATH):
        if not isinstance(item, dict):
            continue

        try:
            rsi = float((item.get("indicators") or {}).get("rsi"))
            mfi = float((item.get("indicators") or {}).get("mfi"))
            actual_price = float(item.get("actualPrice"))
        except Exception:
            continue

        if any(v is None for v in [rsi, mfi, actual_price]):
            continue

        rows.append(
            {
                "symbol": str(item.get("symbol") or "").upper(),
                "rsi": rsi,
                "mfi": mfi,
                "actualPrice": actual_price,
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    ensure_path(PREDICTIONS_PATH, default_content="[]")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_training_frame()
    if df.empty:
        print(json.dumps({"error": "No usable rows found in watchlist_predictions.json"}, ensure_ascii=False, indent=2))
        return 1

    if len(df) < 2:
        print(json.dumps({"error": "Need at least 2 training rows to fit LinearRegression"}, ensure_ascii=False, indent=2))
        return 1

    X = df[["rsi", "mfi"]]
    y = df["actualPrice"]

    test_size = 0.25 if len(df) >= 4 else 0.5
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else None

    payload = {
        "featureNames": ["rsi", "mfi"],
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "trainRows": int(len(X_train)),
        "testRows": int(len(X_test)),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6) if r2 is not None else None,
        "sourceFile": str(PREDICTIONS_PATH),
    }

    joblib.dump(
        {
            "model": model,
            "metadata": payload,
        },
        MODEL_PATH,
    )

    print(json.dumps({"savedModel": str(MODEL_PATH), "metrics": payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
