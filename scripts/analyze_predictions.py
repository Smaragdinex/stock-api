from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "predictions_log.json"
OUTPUT_DIR = BASE_DIR / "reports"


def load_predictions() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()

    try:
        data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    if not isinstance(data, list):
        return pd.DataFrame()

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        actual = item.get("actualPrice")
        predicted = item.get("predictedMid")
        try:
            actual = float(actual)
            predicted = float(predicted)
        except Exception:
            continue

        rows.append(
            {
                "prediction_id": item.get("prediction_id"),
                "symbol": item.get("symbol"),
                "predicted_at": item.get("predicted_at"),
                "actual": actual,
                "predicted": predicted,
                "residual": actual - predicted,
            }
        )

    return pd.DataFrame(rows)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_residual_vs_predicted(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["predicted"], df["residual"], alpha=0.7)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual vs Predicted Price")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_actual_vs_predicted(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(df["predicted"], df["actual"], alpha=0.7)

    min_val = min(df["predicted"].min(), df["actual"].min())
    max_val = max(df["predicted"].max(), df["actual"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label=r"$45^{\circ}$ reference line")

    plt.xlabel("Predicted Price")
    plt.ylabel("Actual Price")
    plt.title("Actual vs Predicted Scatter Plot")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_error_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["residual"], bins=20, alpha=0.85, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Error Histogram")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze stock prediction logs and generate charts.")
    parser.add_argument("--symbol", help="Optional symbol filter, e.g. SNDK or MRVL")
    args = parser.parse_args()

    ensure_output_dir()

    df = load_predictions()
    if df.empty:
        print("No valid prediction data found.")
        return 1

    if args.symbol:
        df = df[df["symbol"].astype(str).str.upper() == args.symbol.upper()]
        if df.empty:
            print(f"No rows found for symbol {args.symbol}.")
            return 1

    rmse = mean_squared_error(df["actual"], df["predicted"], squared=False)
    r2 = r2_score(df["actual"], df["predicted"]) if len(df) >= 2 else None
    mean_residual = float(df["residual"].mean())

    suffix = f"_{args.symbol.upper()}" if args.symbol else ""
    residual_path = OUTPUT_DIR / f"residual_vs_predicted{suffix}.png"
    scatter_path = OUTPUT_DIR / f"actual_vs_predicted{suffix}.png"
    hist_path = OUTPUT_DIR / f"residual_histogram{suffix}.png"

    plot_residual_vs_predicted(df, residual_path)
    plot_actual_vs_predicted(df, scatter_path)
    plot_error_histogram(df, hist_path)

    summary = {
        "sampleSize": int(len(df)),
        "symbol": args.symbol.upper() if args.symbol else None,
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2), 6) if r2 is not None else None,
        "meanResidual": round(mean_residual, 6),
        "files": {
            "residualVsPredicted": str(residual_path),
            "actualVsPredicted": str(scatter_path),
            "residualHistogram": str(hist_path),
        },
    }

    summary_path = OUTPUT_DIR / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
