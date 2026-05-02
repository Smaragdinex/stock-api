from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from analysis_utils import DEFAULT_OUTPUT_DIR, DEFAULT_PREDICTIONS_PATH, DEFAULT_WATCHLIST_SYMBOLS, append_unique_entries, ensure_path


BASE_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = DEFAULT_PREDICTIONS_PATH
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
API_BASE_URL = "https://web-production-f64b.up.railway.app"
WATCHLIST_SYMBOLS = DEFAULT_WATCHLIST_SYMBOLS

WATCHLIST = [
    {
        "symbol": "TSM",
        "companyName": "TSMC",
        "bias": "up",
        "confidence": "high",
        "technicalSummary": "先進製程需求穩，趨勢偏多。",
        "newsImpact": "positive",
        "newsSummary": "AI 晶片需求與先進製程持續支撐。",
        "indicators": {"rsi": 61, "mfi": 60, "signal": "強勢"},
        "marketContext": "watchlist",
    },
    {
        "symbol": "NVDA",
        "companyName": "NVIDIA",
        "bias": "up",
        "confidence": "high",
        "technicalSummary": "AI 伺服器需求延續，強勢趨勢未破。",
        "newsImpact": "positive",
        "newsSummary": "Blackwell 與資料中心需求持續支撐。",
        "indicators": {"rsi": 64, "mfi": 63, "signal": "強勢"},
        "marketContext": "watchlist",
    },
    {
        "symbol": "MRVL",
        "companyName": "Marvell Technology",
        "bias": "up",
        "confidence": "high",
        "technicalSummary": "AI 與資料中心需求仍強，趨勢偏多。",
        "newsImpact": "positive",
        "newsSummary": "Google AI chips 與 hyperscaler 設計案帶來正面動能。",
        "indicators": {"rsi": 66, "mfi": 61, "signal": "強勢"},
        "marketContext": "watchlist",
    },
    {
        "symbol": "STX",
        "companyName": "Seagate Technology",
        "bias": "up",
        "confidence": "medium",
        "technicalSummary": "儲存需求穩定，但高位區間震盪。",
        "newsImpact": "neutral",
        "newsSummary": "AI 與儲存週期支撐，但追價空間有限。",
        "indicators": {"rsi": 58, "mfi": 55, "signal": "中性"},
        "marketContext": "watchlist",
    },
]


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_quote(symbol: str, quote_payload: dict) -> dict:
    return {
        "symbol": symbol.upper(),
        "companyName": quote_payload.get("shortName") or quote_payload.get("longName") or symbol.upper(),
        "currentPrice": quote_payload.get("displayPrice") or quote_payload.get("regularPrice") or quote_payload.get("price"),
    }


def normalize_result(symbol: str, request_payload: dict, response_payload: dict) -> dict:
    return {
        "prediction_id": response_payload.get("predictionId") or response_payload.get("prediction_id") or f"pred_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "symbol": symbol.upper(),
        "companyName": request_payload.get("companyName"),
        "predicted_at": datetime.now().isoformat(),
        "currentPrice": response_payload.get("currentPrice", request_payload.get("currentPrice")),
        "predictedLow": response_payload.get("predictedLow", request_payload.get("predictedLow")),
        "predictedHigh": response_payload.get("predictedHigh", request_payload.get("predictedHigh")),
        "predictedMid": response_payload.get("predictedMid"),
        "predictedDirection": response_payload.get("predictedDirection"),
        "action": response_payload.get("action"),
        "confidence": response_payload.get("confidence"),
        "summary": response_payload.get("summary"),
        "technical": response_payload.get("technical"),
        "sentiment": response_payload.get("sentiment"),
        "watchPoints": response_payload.get("watchPoints"),
        "bias": response_payload.get("bias"),
        "newsImpact": response_payload.get("newsImpact"),
        "newsSummary": response_payload.get("newsSummary"),
        "indicators": request_payload.get("indicators"),
        "marketContext": request_payload.get("marketContext"),
        "source": "railway-api",
    }


def main() -> int:
    ensure_path(PREDICTIONS_PATH, default_content="[]")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    new_entries: list[dict] = []
    print("Fetching and appending predictions...")
    for item in WATCHLIST:
        try:
            quote = get_json(f"{API_BASE_URL}/quote/{item['symbol']}")
            quote_info = normalize_quote(item["symbol"], quote)
            current_price = quote_info.get("currentPrice")
            if current_price is None:
                raise ValueError(f"No current price for {item['symbol']}")

            technical_summary = item["technicalSummary"]
            if quote.get("change") is not None:
                technical_summary = f"{technical_summary} 現價變動 {quote.get('change')}。"

            payload = {
                "symbol": item["symbol"],
                "companyName": quote_info["companyName"],
                "language": "zh-TW",
                "currentPrice": current_price,
                "predictedLow": round(float(current_price) * 0.97, 2),
                "predictedHigh": round(float(current_price) * 1.03, 2),
                "bias": item["bias"],
                "confidence": item["confidence"],
                "technicalSummary": technical_summary,
                "newsImpact": item["newsImpact"],
                "newsSummary": item["newsSummary"],
                "indicators": item["indicators"],
                "marketContext": item["marketContext"],
            }

            response = post_json(f"{API_BASE_URL}/api/ai/analyze", payload)
            entry = normalize_result(item["symbol"], payload, response)
            new_entries.append(entry)
            print(f"OK: {item['symbol']} -> prediction_id={entry['prediction_id']}")
        except (HTTPError, URLError, TimeoutError, Exception) as exc:
            print(f"FAIL: {item['symbol']} -> {exc}")

    append_unique_entries(PREDICTIONS_PATH, new_entries)

    report = {
        "syncedCount": len(new_entries),
        "watchlist": WATCHLIST_SYMBOLS,
        "predictionFile": str(PREDICTIONS_PATH),
        "note": "Predictions only. Use fetch_actuals.py after Monday close.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
