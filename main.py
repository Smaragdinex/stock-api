from fastapi import FastAPI
import yfinance as yf
import pandas as pd

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
        return "regular"

    state = str(market_state).upper()
    if "PRE" in state:
        return "pre"
    if "POST" in state or "AFTER" in state:
        return "post"
    if "REGULAR" in state or "OPEN" in state:
        return "regular"
    return state.lower()


@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        last_price = _safe_float(info.get("lastPrice"))
        previous_close = _safe_float(info.get("previousClose"))
        day_high = _safe_float(info.get("dayHigh"))
        day_low = _safe_float(info.get("dayLow"))
        currency = info.get("currency") or "USD"
        market_state = info.get("marketState")
        session = _normalize_session(market_state)

        intraday = ticker.history(period="2d", interval="1m", prepost=True)
        if intraday.empty:
            if last_price is None:
                return {"error": f"找不到股票代號: {symbol}"}

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
            }

        close_prices = intraday["Close"].dropna()
        if close_prices.empty:
            return {"error": f"無法取得最新報價: {symbol}"}

        display_price = _safe_float(close_prices.iloc[-1])

        regular_only = intraday.between_time("09:30", "16:00")
        regular_close_prices = regular_only["Close"].dropna() if not regular_only.empty else pd.Series(dtype=float)
        regular_price = _safe_float(regular_close_prices.iloc[-1]) if not regular_close_prices.empty else last_price

        extended_only = pd.concat([
            intraday.between_time("04:00", "09:29"),
            intraday.between_time("16:01", "20:00")
        ]).sort_index()
        extended_close_prices = extended_only["Close"].dropna() if not extended_only.empty else pd.Series(dtype=float)
        extended_price = _safe_float(extended_close_prices.iloc[-1]) if not extended_close_prices.empty else None

        if display_price is None:
            display_price = extended_price or regular_price or last_price

        if session == "pre":
            display_price = extended_price or display_price or regular_price or last_price
        elif session == "post":
            display_price = extended_price or display_price or regular_price or last_price
        else:
            display_price = regular_price or display_price or last_price

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
            "5y": "1wk"
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

                result.append({
                    "date": date_str,
                    "price": round(price_val, 2),
                    "ma5": round(ma5_val, 2) if ma5_val is not None else None
                })
            except Exception:
                continue

        return {
            "stock": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": result
        }

    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}
