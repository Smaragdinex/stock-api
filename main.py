from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

from fastapi import FastAPI, Query
import pandas as pd
import twstock
import yfinance as yf
from yahooquery import search as yahoo_search

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TW_STOCKS_PATH = BASE_DIR / "tw_stocks.json"


@app.get("/")
def root():
    return {"message": "API is running"}


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _is_tw_symbol(symbol: str) -> bool:
    upper = symbol.upper()
    return upper.endswith(".TW") or upper.endswith(".TWO")


def _tw_code(symbol: str) -> str:
    return symbol.split(".")[0]


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
    return state.lower()


def _infer_session_from_time(has_pre_price, has_post_price):
    ny_now = datetime.now(ZoneInfo("America/New_York"))
    current_minutes = ny_now.hour * 60 + ny_now.minute
    pre_start = 4 * 60
    regular_start = 9 * 60 + 30
    regular_end = 16 * 60
    post_end = 20 * 60
    if pre_start <= current_minutes < regular_start and has_pre_price:
        return "pre"
    if regular_end < current_minutes <= post_end and has_post_price:
        return "post"
    if regular_start <= current_minutes <= regular_end:
        return "regular"
    return "regular"


def _company_snapshot(ticker):
    info = getattr(ticker, "info", {}) or {}
    fast_info = getattr(ticker, "fast_info", {}) or {}

    company_name = info.get("longName") or info.get("shortName") or info.get("displayName")
    market_cap = _safe_float(info.get("marketCap") or fast_info.get("marketCap"))
    open_price = _safe_float(info.get("open") or fast_info.get("open"))
    fifty_two_week_high = _safe_float(info.get("fiftyTwoWeekHigh") or fast_info.get("yearHigh"))
    fifty_two_week_low = _safe_float(info.get("fiftyTwoWeekLow") or fast_info.get("yearLow"))
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

    return {
        "companyName": company_name,
        "marketCap": market_cap,
        "openPrice": open_price,
        "fiftyTwoWeekHigh": fifty_two_week_high,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "eps": eps,
        "peRatio": pe_ratio,
        "dividendYield": dividend_yield,
    }


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _load_tw_stocks():
    if not TW_STOCKS_PATH.exists():
        return []
    try:
        return json.loads(TW_STOCKS_PATH.read_text())
    except Exception:
        return []


def _search_tw_stocks(query: str, limit: int):
    normalized = query.strip().lower()
    if not normalized:
        return []

    stocks = _load_tw_stocks()
    results = []
    for item in stocks:
        symbol = str(item.get("symbol", ""))
        name = str(item.get("name", ""))
        exchange = item.get("exchange", "TWSE")
        quote_type = item.get("type", "EQUITY")
        haystacks = [symbol.lower(), name.lower()]
        if any(normalized in hay for hay in haystacks):
            results.append({
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "type": quote_type,
            })
        if len(results) >= limit:
            break
    return results


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


def _twstock_quote(symbol: str):
    code = _tw_code(symbol)
    stock = twstock.realtime.get(code)
    if not stock.get("success"):
        return {"error": f"找不到台股代號: {symbol}"}

    realtime = stock.get("realtime", {}) or {}
    info = stock.get("info", {}) or {}
    latest_trade_price = realtime.get("latest_trade_price") or []
    best_bid_price = realtime.get("best_bid_price") or []
    best_ask_price = realtime.get("best_ask_price") or []

    last_price = None
    if latest_trade_price:
        valid = [p for p in latest_trade_price if p not in ("-", "", None)]
        if valid:
            last_price = _safe_float(valid[-1])
    if last_price is None and best_bid_price:
        valid = [p for p in best_bid_price if p not in ("-", "", None)]
        if valid:
            last_price = _safe_float(valid[0])
    if last_price is None and best_ask_price:
        valid = [p for p in best_ask_price if p not in ("-", "", None)]
        if valid:
            last_price = _safe_float(valid[0])

    open_price = _safe_float(realtime.get("open"))
    high_price = _safe_float(realtime.get("high"))
    low_price = _safe_float(realtime.get("low"))
    previous_close = _safe_float(realtime.get("yesterday_close"))
    change = round(last_price - previous_close, 2) if last_price is not None and previous_close is not None else None
    change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None
    name = info.get("name") or symbol

    return {
        "stock": symbol.upper(),
        "price": round(last_price, 2) if last_price is not None else None,
        "displayPrice": round(last_price, 2) if last_price is not None else None,
        "regularPrice": round(last_price, 2) if last_price is not None else None,
        "extendedPrice": None,
        "previousClose": round(previous_close, 2) if previous_close is not None else None,
        "change": change,
        "changePercent": change_percent,
        "dayHigh": round(high_price, 2) if high_price is not None else None,
        "dayLow": round(low_price, 2) if low_price is not None else None,
        "currency": "TWD",
        "marketState": None,
        "session": "regular",
        "companyName": name,
        "marketCap": None,
        "openPrice": round(open_price, 2) if open_price is not None else None,
        "fiftyTwoWeekHigh": None,
        "fiftyTwoWeekLow": None,
        "eps": None,
        "peRatio": None,
        "dividendYield": None,
    }


def _twstock_history(symbol: str, period: str):
    code = _tw_code(symbol)
    stock = twstock.Stock(code)
    history = stock.fetch_from(2024, 1)
    if not history:
        return {"error": f"找不到台股代號: {symbol}"}

    interval_map = {
        "1d": 1,
        "5d": 5,
        "1mo": 22,
        "ytd": 120,
        "3mo": 66,
        "6mo": 132,
        "1y": 264,
        "5y": 1320,
    }
    take = interval_map.get(period, 132)
    sliced = history[-take:]

    closes = [item.close for item in sliced]
    highs = [item.high for item in sliced]
    lows = [item.low for item in sliced]
    volumes = [item.capacity for item in sliced]

    result = []
    for idx, item in enumerate(sliced):
        window = closes[max(0, idx - 4): idx + 1]
        ma5 = round(sum(window) / len(window), 2) if len(window) == 5 else None
        result.append({
            "date": item.date.strftime("%Y-%m-%d"),
            "price": round(item.close, 2),
            "ma5": ma5,
            "high": round(item.high, 2),
            "low": round(item.low, 2),
            "volume": float(item.capacity),
        })

    payload = {
        "stock": symbol.upper(),
        "period": period,
        "interval": "1d",
        "data": result,
    }
    return _attach_indicators(payload, highs, lows, closes, volumes)


@app.get("/search")
def search_symbols(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=20)):
    try:
        seen = set()
        results = []

        if _contains_chinese(q):
            for item in _search_tw_stocks(q, limit):
                symbol = item["symbol"]
                if symbol in seen:
                    continue
                seen.add(symbol)
                results.append(item)

        raw = yahoo_search(q)
        quotes = raw.get("quotes", []) if isinstance(raw, dict) else []
        for item in quotes:
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or item.get("dispSecIndFlag") or symbol
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


@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    try:
        if _is_tw_symbol(symbol):
            return _twstock_quote(symbol)

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        snapshot = _company_snapshot(ticker)

        last_price = _safe_float(info.get("lastPrice"))
        previous_close = _safe_float(info.get("previousClose"))
        day_high = _safe_float(info.get("dayHigh"))
        day_low = _safe_float(info.get("dayLow"))
        currency = info.get("currency") or "USD"
        market_state = info.get("marketState")

        intraday = ticker.history(period="2d", interval="1m", prepost=True)
        if intraday.empty:
            if last_price is None:
                return {"error": f"找不到股票代號: {symbol}"}
            session = _normalize_session(market_state) or "regular"
            change = round(last_price - previous_close, 2) if previous_close is not None else None
            change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None
            return {
                "stock": symbol.upper(),
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
            }

        close_prices = intraday["Close"].dropna()
        if close_prices.empty:
            return {"error": f"無法取得最新報價: {symbol}"}

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
        if session is None:
            session = _infer_session_from_time(pre_price is not None, post_price is not None)

        if session == "pre":
            extended_price = pre_price or latest_intraday_price
            display_price = pre_price or extended_price or latest_intraday_price or last_price or regular_price
        elif session == "post":
            extended_price = post_price or latest_intraday_price
            display_price = post_price or extended_price or latest_intraday_price or last_price or regular_price
        else:
            extended_price = post_price or pre_price
            display_price = regular_price or last_price or latest_intraday_price or extended_price

        if previous_close is None and len(close_prices) > 1:
            previous_close = _safe_float(close_prices.iloc[-2])

        change = round(display_price - previous_close, 2) if previous_close is not None and display_price is not None else None
        change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None

        return {
            "stock": symbol.upper(),
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
        }
    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}


@app.get("/stock/{symbol}")
def get_stock_data(symbol: str, period: str = "6mo"):
    try:
        if _is_tw_symbol(symbol):
            return _twstock_history(symbol, period)

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
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return {"error": f"找不到股票代號: {symbol}"}

        if isinstance(df["Close"], pd.DataFrame):
            close_prices = df["Close"].iloc[:, 0]
            high_prices = df["High"].iloc[:, 0]
            low_prices = df["Low"].iloc[:, 0]
            volume_series = df["Volume"].iloc[:, 0]
        else:
            close_prices = df["Close"]
            high_prices = df["High"]
            low_prices = df["Low"]
            volume_series = df["Volume"]

        ma5 = close_prices.rolling(window=5).mean()
        result = []
        for i in range(len(df)):
            try:
                date_str = str(df.index[i])
                price_val = float(close_prices.iloc[i])
                ma5_val = float(ma5.iloc[i]) if pd.notnull(ma5.iloc[i]) else None
                result.append({
                    "date": date_str,
                    "price": round(price_val, 2),
                    "ma5": round(ma5_val, 2) if ma5_val is not None else None,
                    "high": round(float(high_prices.iloc[i]), 2),
                    "low": round(float(low_prices.iloc[i]), 2),
                    "volume": float(volume_series.iloc[i]) if pd.notnull(volume_series.iloc[i]) else 0.0,
                })
            except Exception:
                continue

        payload = {
            "stock": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": result,
        }
        closes = [float(x) for x in close_prices.tolist()]
        highs = [float(x) for x in high_prices.tolist()]
        lows = [float(x) for x in low_prices.tolist()]
        volumes = [float(x) if pd.notnull(x) else 0.0 for x in volume_series.tolist()]
        return _attach_indicators(payload, highs, lows, closes, volumes)
    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}
