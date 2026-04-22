from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Query
import pandas as pd
import yfinance as yf
from yahooquery import search as yahoo_search

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is running"}


def _safe_float(value):
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


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
    fifty_two_week_high = _safe_float(
        info.get("fiftyTwoWeekHigh")
        or info.get("fiftyTwoWeekHighChangePercent")
        or fast_info.get("yearHigh")
    )
    fifty_two_week_low = _safe_float(
        info.get("fiftyTwoWeekLow")
        or fast_info.get("yearLow")
    )
    eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    pe_ratio = _safe_float(info.get("trailingPE") or info.get("forwardPE"))

    dividend_yield_raw = _safe_float(info.get("dividendYield"))
    if dividend_yield_raw is not None and dividend_yield_raw <= 1:
        dividend_yield = dividend_yield_raw * 100
    else:
        dividend_yield = dividend_yield_raw

    return {
        "companyName": company_name,
        "marketCap": market_cap,
        "fiftyTwoWeekHigh": fifty_two_week_high,
        "fiftyTwoWeekLow": fifty_two_week_low,
        "eps": eps,
        "peRatio": pe_ratio,
        "dividendYield": dividend_yield,
    }


@app.get("/search")
def search_symbols(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=20)):
    try:
        raw = yahoo_search(q)
        quotes = raw.get("quotes", []) if isinstance(raw, dict) else []

        seen = set()
        results = []
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
            results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "type": quote_type,
                }
            )

            if len(results) >= limit:
                break

        return {"query": q, "results": results}
    except Exception as e:
        return {"error": f"搜尋失敗: {str(e)}", "results": []}


@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    try:
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

        extended_price = None
        if session == "pre":
            extended_price = pre_price or latest_intraday_price
        elif session == "post":
            extended_price = post_price or latest_intraday_price
        else:
            extended_price = post_price or pre_price

        display_price = None
        if session == "pre":
            display_price = pre_price or extended_price or latest_intraday_price or last_price or regular_price
        elif session == "post":
            display_price = post_price or extended_price or latest_intraday_price or last_price or regular_price
        else:
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
        else:
            close_prices = df["Close"]

        ma5 = close_prices.rolling(window=5).mean()

        result = []
        for i in range(len(df)):
            try:
                date_str = str(df.index[i])
                price_val = float(close_prices.iloc[i])
                ma5_val = float(ma5.iloc[i]) if pd.notnull(ma5.iloc[i]) else None
                result.append(
                    {
                        "date": date_str,
                        "price": round(price_val, 2),
                        "ma5": round(ma5_val, 2) if ma5_val is not None else None,
                    }
                )
            except Exception:
                continue

        return {
            "stock": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": result,
        }

    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}
