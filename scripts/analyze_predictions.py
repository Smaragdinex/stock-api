from __future__ import annotations

import argparse
import json

from analysis_utils import DEFAULT_OUTPUT_DIR, DEFAULT_PREDICTIONS_PATH, DEFAULT_WATCHLIST_SYMBOLS, build_group_report, ensure_path, load_predictions


PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
WATCHLIST_SYMBOLS = DEFAULT_WATCHLIST_SYMBOLS


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze local prediction logs and generate charts.")
    parser.add_argument("--symbol", help="Optional symbol filter, e.g. SNDK or MRVL")
    parser.add_argument("--all-symbols", nargs="*", default=WATCHLIST_SYMBOLS, help="Generate per-symbol charts for these symbols after the overall charts")
    args = parser.parse_args()

    ensure_path(PREDICTIONS_PATH, default_content="[]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions(PREDICTIONS_PATH)
    if df.empty:
        print("No valid prediction data found.")
        return 1

    outputs = []
    if args.symbol:
        outputs.append(build_group_report(df, OUTPUT_DIR, args.symbol))
    else:
        outputs.append(build_group_report(df, OUTPUT_DIR, None))
        for sym in args.all_symbols:
            outputs.append(build_group_report(df, OUTPUT_DIR, sym))

    print(json.dumps({"generated": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
