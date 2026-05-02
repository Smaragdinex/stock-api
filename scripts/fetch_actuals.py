from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

from analysis_utils import DEFAULT_ACTUALS_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_PREDICTIONS_PATH, ensure_path, load_json_list, save_json_list


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH
ACTUALS_PATH = DEFAULT_ACTUALS_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
WATCHLIST = ["SNDK", "MRVL", "NVDA", "STX", "TSM"]


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _fetch_close_price(symbol: str) -> float | None:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    if hist.empty:
        return None
    return round(float(hist["Close"].iloc[-1]), 2)


def update_actual_prices() -> dict:
    ensure_path(PREDICTIONS_PATH, default_content="[]")
    ensure_path(ACTUALS_PATH, default_content="[]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = load_json_list(PREDICTIONS_PATH)
    if not predictions:
        return {"updatedCount": 0, "message": "No predictions found."}

    updated_count = 0
    updated_predictions = []
    actual_rows = load_json_list(ACTUALS_PATH)
    actual_by_symbol = {str(row.get("symbol") or "").upper(): row for row in actual_rows if isinstance(row, dict) and row.get("symbol")}

    for entry in predictions:
        if not isinstance(entry, dict):
            updated_predictions.append(entry)
            continue

        symbol = str(entry.get("symbol") or "").upper()
        if symbol not in WATCHLIST:
            updated_predictions.append(entry)
            continue

        actual_price = entry.get("actualPrice")
        if actual_price is None:
            try:
                real_price = _fetch_close_price(symbol)
            except Exception as exc:
                print(f"⚠️ 無法抓取 {symbol}: {exc}")
                updated_predictions.append(entry)
                continue

            if real_price is None:
                print(f"⚠️ {symbol} 無法取得收盤價")
                updated_predictions.append(entry)
                continue

            predicted_mid = _safe_float(entry.get("predictedMid"))
            entry["actualPrice"] = real_price
            if predicted_mid is not None:
                entry["residual"] = round(real_price - predicted_mid, 2)
                entry["isCorrect"] = abs(entry["residual"]) < (real_price * 0.02)
            updated_count += 1

            actual_by_symbol[symbol] = {
                "symbol": symbol,
                "date": datetime.now(timezone.utc).date().isoformat(),
                "actualOpen": real_price,
                "actualHigh": real_price,
                "actualLow": real_price,
                "actualClose": real_price,
                "source": "yfinance",
            }

        updated_predictions.append(entry)

    save_json_list(PREDICTIONS_PATH, updated_predictions)
    save_json_list(ACTUALS_PATH, list(actual_by_symbol.values()))

    return {
        "updatedCount": updated_count,
        "predictionsFile": str(PREDICTIONS_PATH),
        "actualsFile": str(ACTUALS_PATH),
    }


def main() -> int:
    result = update_actual_prices()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
