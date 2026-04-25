from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json
from urllib.parse import urlparse

from fastapi import FastAPI, Query
import pandas as pd
import yfinance as yf
from yahooquery import search as yahoo_search

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TW_STOCKS_PATH = BASE_DIR / "tw_stocks.json"

_TW_STOCKS_CACHE = None
_TW_STOCKS_INDEX = None
_WATCHLIST_CACHE = {}
_ENDPOINT_CACHE = {}


@app.get("/")
def root():
    return {"message": "API is running"}


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _load_tw_stocks():
    global _TW_STOCKS_CACHE
    if _TW_STOCKS_CACHE is not None:
        return _TW_STOCKS_CACHE

    if not TW_STOCKS_PATH.exists():
        _TW_STOCKS_CACHE = []
        return _TW_STOCKS_CACHE

    try:
        _TW_STOCKS_CACHE = json.loads(TW_STOCKS_PATH.read_text(encoding="utf-8"))
    except Exception:
        _TW_STOCKS_CACHE = []

    return _TW_STOCKS_CACHE


def _normalize_search_text(text: str) -> str:
    return "".join(str(text or "").strip().lower().split())


def _get_tw_index():
    global _TW_STOCKS_INDEX
    if _TW_STOCKS_INDEX is not None:
        return _TW_STOCKS_INDEX

    stocks = _load_tw_stocks()
    by_code = {}
    by_symbol = {}
    prepared = []

    for item in stocks:
        symbol = str(item.get("symbol", "")).upper()
        code = str(item.get("code") or symbol.split(".")[0]).strip()
        name = str(item.get("name", "")).strip()
        exchange = item.get("exchange", "")
        quote_type = item.get("type", "EQUITY")

        prepared_item = {
            "symbol": symbol,
            "code": code,
            "name": name,
            "exchange": exchange,
            "type": quote_type,
            "nameNormalized": _normalize_search_text(name),
            "symbolNormalized": symbol.lower(),
            "codeNormalized": code.lower(),
        }
        prepared.append(prepared_item)
        by_symbol[symbol] = prepared_item
        by_code[code] = prepared_item

    _TW_STOCKS_INDEX = {
        "prepared": prepared,
        "by_code": by_code,
        "by_symbol": by_symbol,
    }
    return _TW_STOCKS_INDEX


def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def _lookup_tw_symbol(symbol: str):
    normalized = normalize_symbol(symbol)
    if not normalized.isdigit():
        return None

    target = normalized.strip()
    item = _get_tw_index()["by_code"].get(target)
    return item["symbol"] if item else None


def candidate_symbols(symbol: str):
    normalized = normalize_symbol(symbol)

    if "." in normalized:
        return [normalized]

    if normalized.isdigit():
        resolved_tw_symbol = _lookup_tw_symbol(normalized)
        ordered = []

        if resolved_tw_symbol:
            ordered.append(resolved_tw_symbol)

            if resolved_tw_symbol.endswith(".TW"):
                ordered.append(f"{normalized}.TWO")
            elif resolved_tw_symbol.endswith(".TWO"):
                ordered.append(f"{normalized}.TW")
        else:
            ordered.extend([f"{normalized}.TW", f"{normalized}.TWO"])

        ordered.append(normalized)

        deduped = []
        for item in ordered:
            if item not in deduped:
                deduped.append(item)
        return deduped

    return [normalized]


@app.get("/debug_symbol/{symbol}")
def debug_symbol(symbol: str):
    return {
        "input": symbol,
        "normalized": normalize_symbol(symbol),
        "lookup": _lookup_tw_symbol(symbol),
        "candidates": candidate_symbols(symbol),
    }


def _safe_fast_info(ticker):
    try:
        data = getattr(ticker, "fast_info", {}) or {}
        return dict(data)
    except Exception:
        return {}


def _safe_info(ticker):
    try:
        data = getattr(ticker, "info", {}) or {}
        return dict(data)
    except Exception:
        return {}


def _normalize_session(market_state):
    if not market_state:
        return None
    state = str(market_state).upper()
    if "PRE" in state:
        return "pre"
    if "POST" in state or "AFTER" in state:
        return "post"
    if "REGULAR" in state or "OPEN" in state:
        return "regular"
    if "CLOSED" in state or "CLOSE" in state:
        return "closed"
    return state.lower()


def _is_tw_market_symbol(symbol: str) -> bool:
    upper = (symbol or "").upper()
    return upper.endswith(".TW") or upper.endswith(".TWO")


def _market_time_window(symbol: str):
    if _is_tw_market_symbol(symbol):
        return {
            "tz": ZoneInfo("Asia/Taipei"),
            "pre_start": None,
            "regular_start": 9 * 60,
            "regular_end": 13 * 60 + 30,
            "post_end": None,
        }

    return {
        "tz": ZoneInfo("America/New_York"),
        "pre_start": 4 * 60,
        "regular_start": 9 * 60 + 30,
        "regular_end": 16 * 60,
        "post_end": 20 * 60,
    }


def _current_market_phase(symbol: str):
    window = _market_time_window(symbol)
    now = datetime.now(window["tz"])
    current_minutes = now.hour * 60 + now.minute

    pre_start = window["pre_start"]
    regular_start = window["regular_start"]
    regular_end = window["regular_end"]
    post_end = window["post_end"]

    if pre_start is not None and pre_start <= current_minutes < regular_start:
        return "pre"
    if regular_start <= current_minutes <= regular_end:
        return "regular"
    if post_end is not None and regular_end < current_minutes <= post_end:
        return "post"
    return "closed"


def _infer_session_from_time(symbol: str, has_pre_price, has_post_price):
    phase = _current_market_phase(symbol)
    if phase == "pre":
        return "pre" if has_pre_price else "closed"
    if phase == "post":
        return "post" if has_post_price else "closed"
    return phase


def _default_valuation_scenarios():
    return [
        {"id": "bear", "label": "Bear", "revenueGrowthRate": 0.15, "expectedNetMargin": 0.25, "exitPE": 20.0},
        {"id": "base", "label": "Base", "revenueGrowthRate": 0.28, "expectedNetMargin": 0.30, "exitPE": 30.0},
        {"id": "bull", "label": "Bull", "revenueGrowthRate": 0.35, "expectedNetMargin": 0.35, "exitPE": 40.0},
    ]


def _default_tw_eps_pe_scenarios(bucket: str):
    presets = {
        "food": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.95, "targetPE": 12.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.05, "targetPE": 16.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.15, "targetPE": 20.0},
        ],
        "high_end_pcb": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.9, "targetPE": 22.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 27.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.1, "targetPE": 30.0},
        ],
        "electronic_components": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.85, "targetPE": 10.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 14.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.15, "targetPE": 18.0},
        ],
        "semiconductors": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.9, "targetPE": 15.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 20.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.15, "targetPE": 25.0},
        ],
        "shipping": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.7, "targetPE": 6.0},
            {"id": "base", "label": "Base", "epsMultiplier": 0.9, "targetPE": 8.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.1, "targetPE": 10.0},
        ],
        "building_materials": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.85, "targetPE": 10.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 14.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.15, "targetPE": 18.0},
        ],
        "chemicals_materials": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.9, "targetPE": 8.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 10.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.1, "targetPE": 12.0},
        ],
        "financials": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.9, "targetPE": 8.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 10.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.1, "targetPE": 12.0},
        ],
        "default": [
            {"id": "bear", "label": "Bear", "epsMultiplier": 0.9, "targetPE": 10.0},
            {"id": "base", "label": "Base", "epsMultiplier": 1.0, "targetPE": 14.0},
            {"id": "bull", "label": "Bull", "epsMultiplier": 1.15, "targetPE": 18.0},
        ],
    }
    return presets.get(bucket, presets["default"])


def _tw_industry_bucket(symbol, info):
    industry = (info.get("industry") or "").lower()
    sector = (info.get("sector") or "").lower()
    name = (info.get("longName") or info.get("shortName") or "").lower()
    upper_symbol = (symbol or "").upper()

    if upper_symbol in {"2313.TW", "2368.TW", "2383.TW"}:
        return "high_end_pcb"
    if "food" in industry or "packaged foods" in industry or sector == "consumer defensive":
        return "food"
    if "insurance" in industry or sector == "financial services":
        return "financials"
    if "shipping" in industry or "marine shipping" in industry:
        return "shipping"
    if "building materials" in industry or "glass" in name:
        return "building_materials"
    if "chemical" in industry or sector == "basic materials":
        return "chemicals_materials"
    if "semiconductor" in industry:
        return "semiconductors"
    if "electronic component" in industry or "printed circuit" in industry or "computer hardware" in industry or "pcb" in name or "satellite" in name or "hdi" in name:
        return "electronic_components"
    return "default"


def _valuation_overrides(symbol: str):
    overrides = {
        "MRVL": {
            "holdingYears": 3,
            "normalizedRevenuePerShare": 10.7553,
            "notes": "Calibrated normalized revenue/share to align Base case near 203 using 28% growth, 30% net margin, 30x exit P/E over 3 years.",
        }
    }
    return overrides.get((symbol or '').upper(), {})


def _resolved_current_price(info, fast_info):
    return _safe_float(
        (info or {}).get("currentPrice")
        or (info or {}).get("regularMarketPrice")
        or (fast_info or {}).get("lastPrice")
        or (fast_info or {}).get("regularMarketPrice")
    )


def _adr_adjusted_revenue_per_share(symbol, info, fast_info):
    upper = (symbol or "").upper()
    if upper != "TSM":
        return None

    revenue_per_share = _safe_float((info or {}).get("revenuePerShare"))
    shares_outstanding = _safe_float((info or {}).get("sharesOutstanding") or (fast_info or {}).get("shares"))
    total_revenue = _safe_float((info or {}).get("totalRevenue"))

    if revenue_per_share in (None, 0) or shares_outstanding in (None, 0) or total_revenue in (None, 0):
        return None

    implied_shares = total_revenue / revenue_per_share if revenue_per_share not in (None, 0) else None
    if implied_shares in (None, 0):
        return None

    adr_ratio = implied_shares / shares_outstanding
    if adr_ratio <= 1:
        return None

    return revenue_per_share / adr_ratio


def _build_us_valuation_inputs(symbol, info, fast_info):
    overrides = _valuation_overrides(symbol)

    current_price = _resolved_current_price(info, fast_info)
    analyst_target_low = _safe_float(info.get("targetLowPrice"))
    analyst_target_mean = _safe_float(info.get("targetMeanPrice") or info.get("targetMedianPrice"))
    analyst_target_high = _safe_float(info.get("targetHighPrice"))
    analyst_count = int(info.get("numberOfAnalystOpinions") or 0)
    current_revenue_per_share = _adr_adjusted_revenue_per_share(symbol, info, fast_info)
    notes = overrides.get("notes")

    if current_revenue_per_share is None:
        current_revenue_per_share = _safe_float(info.get("revenuePerShare"))

    if current_revenue_per_share is None:
        total_revenue = _safe_float(info.get("totalRevenue"))
        shares_outstanding = _safe_float(info.get("sharesOutstanding") or fast_info.get("shares"))
        if total_revenue not in (None, 0) and shares_outstanding not in (None, 0):
            current_revenue_per_share = total_revenue / shares_outstanding

    if (symbol or '').upper() == 'TSM' and current_revenue_per_share is not None:
        notes = ((notes + " ") if notes else "") + "ADR revenue/share adjusted to avoid mixing Taiwan issuer revenue with ADR share count."

    normalized_revenue_per_share = _safe_float(overrides.get("normalizedRevenuePerShare")) or current_revenue_per_share
    holding_years = int(overrides.get("holdingYears") or 3)

    return {
        "modelType": "revenue_exit_pe",
        "holdingYears": holding_years,
        "currentPrice": current_price,
        "currentRevenuePerShare": current_revenue_per_share,
        "normalizedRevenuePerShare": normalized_revenue_per_share,
        "analystTargetLow": analyst_target_low,
        "analystTargetMean": analyst_target_mean,
        "analystTargetHigh": analyst_target_high,
        "analystCount": analyst_count,
        "notes": notes,
        "isCalibrated": normalized_revenue_per_share != current_revenue_per_share if normalized_revenue_per_share is not None and current_revenue_per_share is not None else False,
    }


def _build_tw_valuation_inputs(symbol, info, fast_info):
    current_price = _safe_float(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or fast_info.get("lastPrice")
        or fast_info.get("regularMarketPrice")
    )
    trailing_eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    forward_eps = _safe_float(info.get("forwardEps"))
    base_eps = forward_eps or trailing_eps
    industry_bucket = _tw_industry_bucket(symbol, info)

    return {
        "modelType": "eps_pe",
        "holdingYears": 1,
        "currentPrice": current_price,
        "baseEPS": base_eps,
        "trailingEPS": trailing_eps,
        "forwardEPS": forward_eps,
        "industryBucket": industry_bucket,
        "analystTargetLow": None,
        "analystTargetMean": None,
        "analystTargetHigh": None,
        "analystCount": 0,
        "notes": f"TW v2.1 model: Target Price = Expected EPS × Target P/E ({industry_bucket.replace('_', ' ')} bucket).",
        "isCalibrated": False,
    }


def _compute_target_price(current_revenue_per_share, revenue_growth_rate, holding_years, expected_net_margin, exit_pe):
    return current_revenue_per_share * pow(1 + revenue_growth_rate, holding_years) * expected_net_margin * exit_pe


def _compute_expected_return(target_price, current_price, holding_years):
    if target_price in (None, 0) or current_price in (None, 0) or holding_years in (None, 0):
        return None
    try:
        return pow(target_price / current_price, 1.0 / holding_years) - 1
    except Exception:
        return None


def _build_us_valuation_payload(symbol, info, fast_info, scenarios=None):
    inputs = _build_us_valuation_inputs(symbol, info, fast_info)
    scenario_defs = scenarios or _default_valuation_scenarios()
    outputs = []

    for scenario in scenario_defs:
        target_price = None
        expected_return = None
        if inputs["normalizedRevenuePerShare"] not in (None, 0):
            target_price = _compute_target_price(
                inputs["normalizedRevenuePerShare"],
                scenario["revenueGrowthRate"],
                inputs["holdingYears"],
                scenario["expectedNetMargin"],
                scenario["exitPE"],
            )
            expected_return = _compute_expected_return(target_price, inputs["currentPrice"], inputs["holdingYears"])

        composite_target_price = None
        if target_price is not None and inputs["analystTargetMean"] is not None:
            composite_target_price = (target_price * 0.4) + (inputs["analystTargetMean"] * 0.6)
        elif target_price is not None:
            composite_target_price = target_price

        outputs.append({
            "id": scenario["id"],
            "label": scenario["label"],
            "revenueGrowthRate": scenario["revenueGrowthRate"],
            "expectedNetMargin": scenario["expectedNetMargin"],
            "exitPE": scenario["exitPE"],
            "targetPE": None,
            "expectedEPS": None,
            "targetPrice": round(target_price, 2) if target_price is not None else None,
            "compositeTargetPrice": round(composite_target_price, 2) if composite_target_price is not None else None,
            "expectedReturn": round(expected_return, 4) if expected_return is not None else None,
        })

    return {
        "stock": symbol.upper(),
        "modelType": inputs["modelType"],
        "holdingYears": inputs["holdingYears"],
        "currentPrice": round(inputs["currentPrice"], 2) if inputs["currentPrice"] is not None else None,
        "currentRevenuePerShare": round(inputs["currentRevenuePerShare"], 4) if inputs["currentRevenuePerShare"] is not None else None,
        "normalizedRevenuePerShare": round(inputs["normalizedRevenuePerShare"], 4) if inputs["normalizedRevenuePerShare"] is not None else None,
        "baseEPS": None,
        "trailingEPS": None,
        "forwardEPS": None,
        "industryBucket": None,
        "analystTargetLow": round(inputs["analystTargetLow"], 2) if inputs["analystTargetLow"] is not None else None,
        "analystTargetMean": round(inputs["analystTargetMean"], 2) if inputs["analystTargetMean"] is not None else None,
        "analystTargetHigh": round(inputs["analystTargetHigh"], 2) if inputs["analystTargetHigh"] is not None else None,
        "analystCount": inputs["analystCount"],
        "isCalibrated": inputs["isCalibrated"],
        "notes": inputs["notes"],
        "scenarios": outputs,
    }


def _build_tw_valuation_payload(symbol, info, fast_info):
    inputs = _build_tw_valuation_inputs(symbol, info, fast_info)
    scenario_defs = _default_tw_eps_pe_scenarios(inputs["industryBucket"])
    outputs = []

    for scenario in scenario_defs:
        target_price = None
        expected_return = None
        expected_eps = None
        if inputs["baseEPS"] not in (None, 0):
            expected_eps = inputs["baseEPS"] * scenario["epsMultiplier"]
            target_price = expected_eps * scenario["targetPE"]
            expected_return = _compute_expected_return(target_price, inputs["currentPrice"], inputs["holdingYears"])

        outputs.append({
            "id": scenario["id"],
            "label": scenario["label"],
            "revenueGrowthRate": None,
            "expectedNetMargin": None,
            "exitPE": None,
            "targetPE": scenario["targetPE"],
            "expectedEPS": round(expected_eps, 2) if expected_eps is not None else None,
            "targetPrice": round(target_price, 2) if target_price is not None else None,
            "compositeTargetPrice": None,
            "expectedReturn": round(expected_return, 4) if expected_return is not None else None,
        })

    return {
        "stock": symbol.upper(),
        "modelType": inputs["modelType"],
        "holdingYears": inputs["holdingYears"],
        "currentPrice": round(inputs["currentPrice"], 2) if inputs["currentPrice"] is not None else None,
        "currentRevenuePerShare": None,
        "normalizedRevenuePerShare": None,
        "baseEPS": round(inputs["baseEPS"], 2) if inputs["baseEPS"] is not None else None,
        "trailingEPS": round(inputs["trailingEPS"], 2) if inputs["trailingEPS"] is not None else None,
        "forwardEPS": round(inputs["forwardEPS"], 2) if inputs["forwardEPS"] is not None else None,
        "industryBucket": inputs["industryBucket"],
        "analystTargetLow": None,
        "analystTargetMean": None,
        "analystTargetHigh": None,
        "analystCount": 0,
        "isCalibrated": inputs["isCalibrated"],
        "notes": inputs["notes"],
        "scenarios": outputs,
    }


def _build_valuation_payload(symbol, info, fast_info, scenarios=None):
    if _is_tw_market_symbol(symbol):
        return _build_tw_valuation_payload(symbol, info, fast_info)
    return _build_us_valuation_payload(symbol, info, fast_info, scenarios=scenarios)


def _company_snapshot(ticker):
    info = _safe_info(ticker)
    fast_info = _safe_fast_info(ticker)

    company_name = info.get("longName") or info.get("shortName") or info.get("displayName")
    market_cap = _safe_float(info.get("marketCap") or fast_info.get("marketCap"))
    open_price = _safe_float(info.get("open") or fast_info.get("open"))
    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh") or info.get("yearHigh"))
    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow") or info.get("yearLow"))
    eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    pe_ratio = _safe_float(info.get("trailingPE") or info.get("forwardPE"))

    dividend_yield_raw = _safe_float(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
    if dividend_yield_raw is None:
        dividend_rate = _safe_float(info.get("dividendRate") or info.get("trailingAnnualDividendRate"))
        reference_price = _safe_float(info.get("previousClose") or info.get("currentPrice") or fast_info.get("lastPrice"))
        if dividend_rate is not None and reference_price not in (None, 0):
            dividend_yield = (dividend_rate / reference_price) * 100
        else:
            dividend_yield = None
    elif dividend_yield_raw < 0.01:
        dividend_yield = dividend_yield_raw * 100
    else:
        dividend_yield = dividend_yield_raw

    valuation = _build_valuation_payload(ticker.ticker if hasattr(ticker, "ticker") else (company_name or "UNKNOWN"), info, fast_info)

    return {
        "companyName": company_name,
        "marketCap": market_cap,
        "openPrice": open_price,
        "fiftyTwoWeekHigh": fifty_two_week_high,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "eps": eps,
        "peRatio": pe_ratio,
        "dividendYield": dividend_yield,
        "valuation": valuation,
    }


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _search_tw_stocks(query: str, limit: int):
    normalized = _normalize_search_text(query)
    if not normalized:
        return []

    prepared = _get_tw_index()["prepared"]
    scored = []

    for item in prepared:
        score = None
        code = item["codeNormalized"]
        symbol = item["symbolNormalized"]
        name = item["nameNormalized"]

        if normalized == code:
            score = 0
        elif normalized == symbol:
            score = 1
        elif code.startswith(normalized):
            score = 2
        elif symbol.startswith(normalized):
            score = 3
        elif name.startswith(normalized):
            score = 4
        elif normalized in name:
            score = 5
        elif normalized in code or normalized in symbol:
            score = 6

        if score is not None:
            scored.append((score, len(item["code"]), item["code"], {
                "symbol": item["symbol"],
                "name": item["name"],
                "exchange": item["exchange"],
                "type": item["type"],
            }))

    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    return [entry[-1] for entry in scored[:limit]]


def _is_tw_symbol(symbol: str) -> bool:
    upper = (symbol or "").upper()
    return upper.endswith(".TW") or upper.endswith(".TWO")


def _tw_search_from_yahoo(query: str, limit: int):
    raw = yahoo_search(query)
    quotes = raw.get("quotes", []) if isinstance(raw, dict) else []
    results = []

    for item in quotes:
        symbol = item.get("symbol")
        if not symbol or not _is_tw_symbol(symbol):
            continue

        quote_type = item.get("quoteType") or ""
        if quote_type and quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}:
            continue

        results.append({
            "symbol": symbol,
            "name": item.get("shortname") or item.get("longname") or symbol,
            "exchange": item.get("exchange") or item.get("exchDisp") or "",
            "type": quote_type,
        })

        if len(results) >= limit:
            break

    return results


def _looks_like_tw_query(query: str) -> bool:
    stripped = query.strip()
    return _contains_chinese(stripped) or stripped.isdigit() or stripped.upper().endswith(".TW") or stripped.upper().endswith(".TWO")


def _compute_rsi(closes, period=14):
    if len(closes) <= period:
        return None

    series = pd.Series(closes, dtype=float)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]

    if pd.isna(last_gain) or pd.isna(last_loss):
        return None
    if last_loss == 0:
        return 100.0

    rs = last_gain / last_loss
    return round(100 - (100 / (1 + rs)), 2)


def _compute_mfi(highs, lows, closes, volumes, period=14):
    if min(len(highs), len(lows), len(closes), len(volumes)) <= period:
        return None

    df = pd.DataFrame({
        "high": pd.Series(highs, dtype=float),
        "low": pd.Series(lows, dtype=float),
        "close": pd.Series(closes, dtype=float),
        "volume": pd.Series(volumes, dtype=float),
    })

    typical = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical * df["volume"]
    direction = typical.diff()

    positive_flow = money_flow.where(direction > 0, 0.0)
    negative_flow = money_flow.where(direction < 0, 0.0).abs()

    pos_sum = positive_flow.rolling(window=period).sum().iloc[-1]
    neg_sum = negative_flow.rolling(window=period).sum().iloc[-1]

    if pd.isna(pos_sum) or pd.isna(neg_sum):
        return None
    if neg_sum == 0:
        return 100.0

    mfr = pos_sum / neg_sum
    return round(100 - (100 / (1 + mfr)), 2)


def _resolve_list_price(symbol: str, quote_payload: dict):
    is_tw_symbol = _is_tw_market_symbol(symbol)
    if is_tw_symbol:
        return quote_payload.get("displayPrice") or quote_payload.get("regularPrice") or quote_payload.get("price")
    return quote_payload.get("regularPrice") or quote_payload.get("displayPrice") or quote_payload.get("price")


def _fetch_watchlist_item(symbol: str):
    for resolved_symbol in candidate_symbols(symbol):
        try:
            quote_payload = get_quote(resolved_symbol)
            if isinstance(quote_payload, dict) and quote_payload.get("error"):
                continue

            stock_payload = get_stock_data(resolved_symbol, period="1d")
            sparkline = stock_payload.get("data", []) if isinstance(stock_payload, dict) else []
            sparkline_prices = [item.get("price") for item in sparkline if item.get("price") is not None]

            company_name = quote_payload.get("companyName") or resolved_symbol.upper()
            list_price = _resolve_list_price(resolved_symbol, quote_payload)

            return {
                "symbol": resolved_symbol.upper(),
                "name": company_name,
                "price": round(list_price, 2) if list_price is not None else None,
                "previousClose": quote_payload.get("previousClose"),
                "change": quote_payload.get("change"),
                "sparkline": sparkline_prices,
            }
        except Exception:
            continue

    return {
        "symbol": normalize_symbol(symbol),
        "name": normalize_symbol(symbol),
        "price": None,
        "previousClose": None,
        "change": None,
        "sparkline": [],
    }


def _cached_watchlist_response(symbols: list[str]):
    now = datetime.now().timestamp()
    ttl_seconds = 20
    normalized_key = tuple(symbols)
    cached = _WATCHLIST_CACHE.get(normalized_key)

    if cached and (now - cached["ts"]) < ttl_seconds:
        return cached["payload"]

    payload = {
        "symbols": symbols,
        "items": [_fetch_watchlist_item(symbol) for symbol in symbols],
    }
    _WATCHLIST_CACHE[normalized_key] = {
        "ts": now,
        "payload": payload,
    }
    return payload


def _cache_get(bucket: str, key, ttl_seconds: int):
    now = datetime.now().timestamp()
    cached = _ENDPOINT_CACHE.get((bucket, key))
    if cached and (now - cached["ts"]) < ttl_seconds:
        return cached["payload"]
    return None


def _cache_set(bucket: str, key, payload):
    _ENDPOINT_CACHE[(bucket, key)] = {
        "ts": datetime.now().timestamp(),
        "payload": payload,
    }
    return payload


def _signal_from_indicators(rsi, mfi):
    if rsi is None and mfi is None:
        return "-"
    if rsi is not None and mfi is not None:
        if rsi >= 60 and mfi >= 60:
            return "強勢"
        if rsi <= 45 and mfi <= 45:
            return "主力出貨"
        return "中性"
    if rsi is not None:
        return "強勢" if rsi >= 60 else "主力出貨" if rsi <= 45 else "中性"
    return "強勢" if mfi >= 60 else "主力出貨" if mfi <= 45 else "中性"


def _attach_indicators(payload, highs, lows, closes, volumes):
    rsi = _compute_rsi(closes)
    mfi = _compute_mfi(highs, lows, closes, volumes)
    payload["rsi"] = rsi
    payload["mfi"] = mfi
    payload["signal"] = _signal_from_indicators(rsi, mfi)
    return payload


def _parse_news_item(item):
    content = item.get("content", {}) if isinstance(item, dict) else {}
    title = content.get("title") or item.get("title")
    if not title:
        return None

    raw_url = (
        (content.get("clickThroughUrl") or {}).get("url")
        or (content.get("canonicalUrl") or {}).get("url")
        or item.get("link")
    )
    provider = (content.get("provider") or {}).get("displayName") or ""
    published_at = content.get("pubDate") or content.get("displayTime") or ""
    summary = content.get("summary") or content.get("description") or ""
    thumbnail = ((content.get("thumbnail") or {}).get("resolutions") or [{}])
    image_url = thumbnail[-1].get("url") if thumbnail else None

    domain = ""
    if raw_url:
        try:
            domain = urlparse(raw_url).netloc
        except Exception:
            domain = ""

    return {
        "id": content.get("id") or item.get("id") or title,
        "title": title,
        "summary": summary,
        "url": raw_url,
        "provider": provider,
        "publishedAt": published_at,
        "imageUrl": image_url,
        "domain": domain,
    }


@app.get("/search")
def search_symbols(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=20)):
    try:
        seen = set()
        results = []
        query = q.strip()
        is_tw_query = _looks_like_tw_query(query)

        if is_tw_query:
            local_results = _search_tw_stocks(query, limit)
            for item in local_results:
                symbol = item["symbol"]
                if symbol in seen:
                    continue
                seen.add(symbol)
                results.append(item)

            # 台股查詢優先本地清單；只有完全沒命中時才補打 Yahoo
            if results:
                return {"query": q, "results": results[:limit]}

            for item in _tw_search_from_yahoo(query, limit):
                symbol = item["symbol"]
                if symbol in seen:
                    continue
                seen.add(symbol)
                results.append(item)

            if results:
                return {"query": q, "results": results[:limit]}

        raw = yahoo_search(query)
        quotes = raw.get("quotes", []) if isinstance(raw, dict) else []

        for item in quotes:
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or symbol
            exchange = item.get("exchange") or item.get("exchDisp") or ""
            quote_type = item.get("quoteType") or ""

            if not symbol or symbol in seen:
                continue
            if quote_type and quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}:
                continue

            seen.add(symbol)
            results.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "type": quote_type,
            })

            if len(results) >= limit:
                break

        return {"query": q, "results": results[:limit]}
    except Exception as e:
        return {"error": f"搜尋失敗: {str(e)}", "results": []}


@app.get("/watchlist")
def get_watchlist(symbols: str = Query(..., min_length=1)):
    requested = [normalize_symbol(item) for item in symbols.split(",") if item.strip()]
    deduped = []
    for item in requested:
        if item not in deduped:
            deduped.append(item)

    return _cached_watchlist_response(deduped)


@app.get("/news/{symbol}")
def get_news(symbol: str, limit: int = Query(10, ge=1, le=30)):
    cache_key = (normalize_symbol(symbol), int(limit))
    cached = _cache_get("news", cache_key, 300)
    if cached is not None:
        return cached

    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)
            raw_news = getattr(ticker, "news", None) or []
            normalized = []

            for item in raw_news:
                parsed = _parse_news_item(item)
                if parsed and parsed.get("url"):
                    normalized.append(parsed)

            normalized.sort(key=lambda x: x.get("publishedAt") or "", reverse=True)

            return _cache_set("news", cache_key, {
                "stock": resolved_symbol.upper(),
                "items": normalized[:limit],
            })
        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("news", cache_key, {"stock": normalize_symbol(symbol), "items": [], "error": last_error})


@app.get("/ratings/{symbol}")
def get_ratings(symbol: str):
    cache_key = normalize_symbol(symbol)
    cached = _cache_get("ratings", cache_key, 300)
    if cached is not None:
        return cached

    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)
            summary = getattr(ticker, "recommendations_summary", None)
            info = _safe_info(ticker)

            latest = None
            if summary is not None and not summary.empty:
                latest = summary.iloc[0].to_dict()

            if not latest:
                latest = {
                    "strongBuy": 0,
                    "buy": 0,
                    "hold": 0,
                    "sell": 0,
                    "strongSell": 0,
                }

            strong_buy = int(latest.get("strongBuy") or 0)
            buy = int(latest.get("buy") or 0)
            hold = int(latest.get("hold") or 0)
            sell = int(latest.get("sell") or 0)
            strong_sell = int(latest.get("strongSell") or 0)
            total = strong_buy + buy + hold + sell + strong_sell

            return _cache_set("ratings", cache_key, {
                "stock": resolved_symbol.upper(),
                "strongBuy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strongSell": strong_sell,
                "total": total,
                "recommendationKey": info.get("recommendationKey"),
                "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            })
        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("ratings", cache_key, {
        "stock": normalize_symbol(symbol),
        "strongBuy": 0,
        "buy": 0,
        "hold": 0,
        "sell": 0,
        "strongSell": 0,
        "total": 0,
        "error": last_error,
    })


@app.get("/earnings/{symbol}")
def get_earnings(symbol: str, limit: int = Query(5, ge=1, le=8)):
    cache_key = (normalize_symbol(symbol), int(limit))
    cached = _cache_get("earnings", cache_key, 300)
    if cached is not None:
        return cached

    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)
            earnings_dates = getattr(ticker, "earnings_dates", None)
            calendar = getattr(ticker, "calendar", None) or {}
            items = []

            if earnings_dates is not None and not earnings_dates.empty:
                df = earnings_dates.head(limit).reset_index()
                for _, row in df.iterrows():
                    earnings_date = row.get("Earnings Date")
                    quarter_label = ""
                    fiscal_label = ""

                    if pd.notnull(earnings_date):
                        ts = pd.Timestamp(earnings_date)
                        quarter = ((ts.month - 1) // 3) + 1
                        quarter_label = f"Q{quarter}"
                        fiscal_label = f"FY{str(ts.year)[-2:]}"

                    estimate = _safe_float(row.get("EPS Estimate"))
                    actual = _safe_float(row.get("Reported EPS"))
                    surprise_pct = _safe_float(row.get("Surprise(%)"))

                    if estimate is not None and pd.isna(estimate):
                        estimate = None
                    if actual is not None and pd.isna(actual):
                        actual = None
                    if surprise_pct is not None and pd.isna(surprise_pct):
                        surprise_pct = None

                    items.append({
                        "quarter": quarter_label,
                        "fiscalYear": fiscal_label,
                        "estimate": round(estimate, 2) if estimate is not None else None,
                        "actual": round(actual, 2) if actual is not None else None,
                        "surprisePercent": round(surprise_pct, 2) if surprise_pct is not None else None,
                        "earningsDate": ts.isoformat() if pd.notnull(earnings_date) else None,
                    })

            next_earnings_date = None
            earnings_timing = None
            cal_date = None
            if isinstance(calendar, dict):
                raw_dates = calendar.get("Earnings Date")
                if isinstance(raw_dates, list) and raw_dates:
                    cal_date = raw_dates[0]
                elif raw_dates:
                    cal_date = raw_dates

            if cal_date:
                if isinstance(cal_date, datetime):
                    next_earnings_date = cal_date.date().isoformat()
                    earnings_timing = "after-hours" if cal_date.hour >= 16 else "before-open" if cal_date.hour < 9 else None
                else:
                    try:
                        next_earnings_date = pd.Timestamp(cal_date).date().isoformat()
                    except Exception:
                        next_earnings_date = str(cal_date)

            return _cache_set("earnings", cache_key, {
                "stock": resolved_symbol.upper(),
                "items": items,
                "nextEarningsDate": next_earnings_date,
                "earningsTiming": earnings_timing,
            })
        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("earnings", cache_key, {"stock": normalize_symbol(symbol), "items": [], "error": last_error})


@app.get("/valuation/{symbol}")
def get_valuation(symbol: str):
    cache_key = normalize_symbol(symbol)
    cached = _cache_get("valuation", cache_key, 300)
    if cached is not None:
        return cached

    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)
            info = _safe_info(ticker)
            fast_info = _safe_fast_info(ticker)
            payload = _build_valuation_payload(resolved_symbol, info, fast_info)

            if payload.get("modelType") == "revenue_exit_pe" and payload.get("normalizedRevenuePerShare") is None:
                payload["error"] = "Valuation inputs unavailable"
            elif payload.get("modelType") == "eps_pe" and payload.get("baseEPS") is None:
                payload["error"] = "Valuation inputs unavailable"
            return _cache_set("valuation", cache_key, payload)
        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("valuation", cache_key, {
        "stock": normalize_symbol(symbol),
        "holdingYears": 3,
        "currentPrice": None,
        "currentRevenuePerShare": None,
        "normalizedRevenuePerShare": None,
        "isCalibrated": False,
        "notes": None,
        "scenarios": [],
        "error": last_error or "Valuation inputs unavailable",
    })


@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    cache_key = normalize_symbol(symbol)
    cached = _cache_get("quote", cache_key, 15)
    if cached is not None:
        return cached

    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)
            info = _safe_fast_info(ticker)
            snapshot = _company_snapshot(ticker)

            last_price = _safe_float(info.get("lastPrice"))
            previous_close = _safe_float(info.get("previousClose"))
            day_high = _safe_float(info.get("dayHigh"))
            day_low = _safe_float(info.get("dayLow"))
            currency = info.get("currency") or "USD"
            market_state = info.get("marketState")

            try:
                intraday = ticker.history(period="2d", interval="1m", prepost=True)
            except Exception:
                intraday = pd.DataFrame()

            if intraday.empty:
                if last_price is None:
                    last_error = f"找不到股票代號: {resolved_symbol}"
                    continue

                session = _normalize_session(market_state)
                if session in (None, "closed"):
                    session = _infer_session_from_time(resolved_symbol, False, False)
                change = round(last_price - previous_close, 2) if previous_close is not None else None
                change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None

                return _cache_set("quote", cache_key, {
                    "stock": resolved_symbol.upper(),
                    "price": round(last_price, 2),
                    "displayPrice": round(last_price, 2),
                    "regularPrice": round(last_price, 2),
                    "extendedPrice": None,
                    "previousClose": round(previous_close, 2) if previous_close is not None else None,
                    "change": change,
                    "changePercent": change_percent,
                    "dayHigh": round(day_high, 2) if day_high is not None else None,
                    "dayLow": round(day_low, 2) if day_low is not None else None,
                    "currency": currency,
                    "marketState": market_state,
                    "session": session,
                    **snapshot,
                })

            close_prices = intraday["Close"].dropna()
            if close_prices.empty:
                last_error = f"無法取得最新報價: {resolved_symbol}"
                continue

            latest_intraday_price = _safe_float(close_prices.iloc[-1])

            regular_only = intraday.between_time("09:30", "16:00")
            regular_close_prices = regular_only["Close"].dropna() if not regular_only.empty else pd.Series(dtype=float)
            regular_price = _safe_float(regular_close_prices.iloc[-1]) if not regular_close_prices.empty else last_price

            pre_only = intraday.between_time("04:00", "09:29")
            pre_close_prices = pre_only["Close"].dropna() if not pre_only.empty else pd.Series(dtype=float)
            pre_price = _safe_float(pre_close_prices.iloc[-1]) if not pre_close_prices.empty else None

            post_only = intraday.between_time("16:01", "20:00")
            post_close_prices = post_only["Close"].dropna() if not post_only.empty else pd.Series(dtype=float)
            post_price = _safe_float(post_close_prices.iloc[-1]) if not post_close_prices.empty else None

            session = _normalize_session(market_state)
            market_phase = _current_market_phase(resolved_symbol)
            inferred_session = _infer_session_from_time(resolved_symbol, pre_price is not None, post_price is not None)

            if session in (None, "closed"):
                session = inferred_session
            elif session == "regular" and inferred_session in {"pre", "post"}:
                session = inferred_session

            if _is_tw_market_symbol(resolved_symbol):
                extended_price = None
                display_price = regular_price or last_price or latest_intraday_price
            else:
                if market_phase == "pre":
                    session = "pre"
                    extended_price = pre_price or latest_intraday_price or last_price
                    display_price = extended_price or regular_price
                elif market_phase == "post":
                    session = "post"
                    extended_price = post_price or latest_intraday_price or last_price
                    display_price = extended_price or regular_price
                elif session == "pre":
                    extended_price = pre_price or latest_intraday_price or last_price
                    display_price = extended_price or regular_price
                elif session == "post":
                    extended_price = post_price or latest_intraday_price or last_price
                    display_price = extended_price or regular_price
                else:
                    extended_price = post_price if session == "closed" else (post_price or pre_price)
                    display_price = regular_price or last_price or latest_intraday_price or extended_price

            if previous_close is None and len(close_prices) > 1:
                previous_close = _safe_float(close_prices.iloc[-2])

            change = round(display_price - previous_close, 2) if previous_close is not None and display_price is not None else None
            change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None

            return _cache_set("quote", cache_key, {
                "stock": resolved_symbol.upper(),
                "price": round(display_price, 2) if display_price is not None else None,
                "displayPrice": round(display_price, 2) if display_price is not None else None,
                "regularPrice": round(regular_price, 2) if regular_price is not None else None,
                "extendedPrice": round(extended_price, 2) if extended_price is not None else None,
                "previousClose": round(previous_close, 2) if previous_close is not None else None,
                "change": change,
                "changePercent": change_percent,
                "dayHigh": round(day_high, 2) if day_high is not None else None,
                "dayLow": round(day_low, 2) if day_low is not None else None,
                "currency": currency,
                "marketState": market_state,
                "session": session,
                **snapshot,
            })
        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("quote", cache_key, {"error": last_error or f"找不到股票代號: {symbol}"})


@app.get("/stock/{symbol}")
def get_stock_data(symbol: str, period: str = "6mo"):
    cache_key = (normalize_symbol(symbol), period)
    cached = _cache_get("stock", cache_key, 20)
    if cached is not None:
        return cached

    allowed_periods = ["1d", "5d", "1mo", "ytd", "3mo", "6mo", "1y", "5y"]
    if period not in allowed_periods:
        return {"error": f"不支援的 period: {period}"}

    interval_map = {
        "1d": "5m",
        "5d": "30m",
        "1mo": "1d",
        "ytd": "1d",
        "3mo": "1d",
        "6mo": "1d",
        "1y": "1d",
        "5y": "1wk",
    }
    interval = interval_map.get(period, "1d")
    last_error = None

    for resolved_symbol in candidate_symbols(symbol):
        try:
            ticker = yf.Ticker(resolved_symbol)

            df = pd.DataFrame()
            history_errors = []
            for auto_adjust in (True, False):
                try:
                    df = ticker.history(period=period, interval=interval, auto_adjust=auto_adjust)
                    if not df.empty:
                        break
                except Exception as history_error:
                    history_errors.append(str(history_error))
                    df = pd.DataFrame()

            if df.empty:
                last_error = history_errors[-1] if history_errors else f"找不到股票代號: {resolved_symbol}"
                continue

            close_prices = df["Close"]
            high_prices = df["High"]
            low_prices = df["Low"]
            volume_series = df["Volume"]
            ma5 = close_prices.rolling(window=5).mean()

            result = []
            for i in range(len(df)):
                try:
                    ts = df.index[i]

                    if period in ["1d", "5d"]:
                        date_str = ts.strftime("%Y-%m-%d %H:%M")
                        chart_label = ts.strftime("%m-%d %H:%M")
                    else:
                        date_str = ts.strftime("%Y-%m-%d")
                        chart_label = ts.strftime("%m-%d")

                    if pd.isnull(close_prices.iloc[i]) or pd.isnull(high_prices.iloc[i]) or pd.isnull(low_prices.iloc[i]):
                        continue

                    price_val = float(close_prices.iloc[i])
                    ma5_val = float(ma5.iloc[i]) if pd.notnull(ma5.iloc[i]) else None
                    high_val = float(high_prices.iloc[i])
                    low_val = float(low_prices.iloc[i])

                    result.append({
                        "date": date_str,
                        "chartLabel": chart_label,
                        "price": round(price_val, 2),
                        "ma5": round(ma5_val, 2) if ma5_val is not None else None,
                        "high": round(high_val, 2),
                        "low": round(low_val, 2),
                        "volume": float(volume_series.iloc[i]) if pd.notnull(volume_series.iloc[i]) else 0.0,
                    })
                except Exception:
                    continue

            payload = {
                "stock": resolved_symbol.upper(),
                "period": period,
                "interval": interval,
                "data": result,
            }

            closes = [float(x) for x in close_prices.tolist()]
            highs = [float(x) for x in high_prices.tolist()]
            lows = [float(x) for x in low_prices.tolist()]
            volumes = [float(x) if pd.notnull(x) else 0.0 for x in volume_series.tolist()]

            return _cache_set("stock", cache_key, _attach_indicators(payload, highs, lows, closes, volumes))

        except Exception as e:
            last_error = str(e)
            continue

    return _cache_set("stock", cache_key, {"error": last_error or f"找不到股票代號: {symbol}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)