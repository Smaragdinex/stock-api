from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTIONS_PATH = BASE_DIR / "watchlist_predictions.json"
DEFAULT_ACTUALS_PATH = BASE_DIR / "watchlist_actuals.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports"
DEFAULT_WATCHLIST_SYMBOLS = ["SNDK", "MRVL", "NVDA", "STX", "TSM"]


def ensure_path(path: Path, default_content: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if default_content is not None and not path.exists():
        path.write_text(default_content, encoding="utf-8")


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_json_list(path: Path, entries: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def append_unique_entries(path: Path, new_entries: list[dict[str, Any]]) -> None:
    existing = load_json_list(path)
    existing_ids = {str(item.get("prediction_id")) for item in existing if isinstance(item, dict) and item.get("prediction_id")}
    for entry in new_entries:
        prediction_id = str(entry.get("prediction_id") or "")
        if prediction_id and prediction_id not in existing_ids:
            existing.append(entry)
            existing_ids.add(prediction_id)
    save_json_list(path, existing)


def load_predictions(path: Path = DEFAULT_PREDICTIONS_PATH) -> pd.DataFrame:
    rows = []
    for item in load_json_list(path):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        predicted_mid = item.get("predictedMid")
        current_price = item.get("currentPrice")
        try:
            predicted_mid = float(predicted_mid)
            current_price = float(current_price)
        except Exception:
            continue
        rows.append(
            {
                "prediction_id": item.get("prediction_id"),
                "symbol": symbol,
                "predicted_at": item.get("predicted_at"),
                "actual": current_price,
                "predicted": predicted_mid,
                "currentPrice": current_price,
                "predictedMid": predicted_mid,
                "predictedLow": item.get("predictedLow"),
                "predictedHigh": item.get("predictedHigh"),
            }
        )
    return pd.DataFrame(rows)


def load_actuals(path: Path = DEFAULT_ACTUALS_PATH) -> pd.DataFrame:
    rows = []
    for item in load_json_list(path):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").upper()
        try:
            actual_open = float(item.get("actualOpen")) if item.get("actualOpen") is not None else None
            actual_high = float(item.get("actualHigh")) if item.get("actualHigh") is not None else None
            actual_low = float(item.get("actualLow")) if item.get("actualLow") is not None else None
            actual_close = float(item.get("actualClose")) if item.get("actualClose") is not None else None
        except Exception:
            continue
        rows.append(
            {
                "symbol": symbol,
                "date": item.get("date"),
                "actualOpen": actual_open,
                "actualHigh": actual_high,
                "actualLow": actual_low,
                "actualClose": actual_close,
            }
        )
    return pd.DataFrame(rows)


def plot_residual_vs_predicted(df: pd.DataFrame, out_path: Path, x_col: str = "predictedMid", y_col: str = "residual") -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual vs Predicted Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_actual_vs_predicted(df: pd.DataFrame, out_path: Path, actual_col: str = "actual", predicted_col: str = "predicted") -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df[predicted_col], df[actual_col], alpha=0.7)
    min_val = min(df[predicted_col].min(), df[actual_col].min())
    max_val = max(df[predicted_col].max(), df[actual_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label=r"$45^{\circ}$ reference line")
    plt.xlabel("Predicted Price")
    plt.ylabel("Actual Price")
    plt.title("Actual vs Predicted Scatter Plot")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_histogram(df: pd.DataFrame, out_path: Path, residual_col: str = "residual") -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df[residual_col], bins=20, alpha=0.85, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Histogram")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_group_report(df: pd.DataFrame, output_dir: Path, symbol: str | None = None) -> dict[str, Any]:
    group = df
    suffix = ""
    if symbol:
        group = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
        suffix = f"_{symbol.upper()}"

    if group.empty:
        return {"symbol": symbol.upper() if symbol else None, "sampleSize": 0, "skipped": True}

    residual = group["actual"] - group["predicted"]
    mse = mean_squared_error(group["actual"], group["predicted"])
    rmse = mse ** 0.5
    r2 = r2_score(group["actual"], group["predicted"]) if len(group) >= 2 else None
    mean_residual = float(residual.mean())

    residual_path = output_dir / f"residual_vs_predicted{suffix}.png"
    scatter_path = output_dir / f"actual_vs_predicted{suffix}.png"
    hist_path = output_dir / f"residual_histogram{suffix}.png"

    plot_df = pd.DataFrame({"predicted": group["predicted"].values, "residual": residual.values})
    plot_residual_vs_predicted(plot_df, residual_path, x_col="predicted", y_col="residual")
    plot_actual_vs_predicted(group, scatter_path, actual_col="actual", predicted_col="predicted")
    plot_error_histogram(plot_df, hist_path, residual_col="residual")

    summary = {
        "sampleSize": int(len(group)),
        "symbol": symbol.upper() if symbol else None,
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6) if r2 is not None else None,
        "meanResidual": round(mean_residual, 6),
        "files": {
            "residualVsPredicted": str(residual_path),
            "actualVsPredicted": str(scatter_path),
            "residualHistogram": str(hist_path),
        },
    }
    (output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
