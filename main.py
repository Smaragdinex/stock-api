from fastapi import FastAPI
import yfinance as yf
import pandas as pd

app = FastAPI()


@app.get("/")
def root():
    return {"message": "API is running"}


@app.get("/quote/{symbol}")
def get_quote(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        last_price = info.get("lastPrice")
        previous_close = info.get("previousClose")
        day_high = info.get("dayHigh")
        day_low = info.get("dayLow")
        currency = info.get("currency") or "USD"
        market_state = info.get("marketState")

        if last_price is None:
            history = ticker.history(period="2d", interval="1m")
            if history.empty:
                return {"error": f"找不到股票代號: {symbol}"}

            close_prices = history["Close"].dropna()
            if close_prices.empty:
                return {"error": f"無法取得最新報價: {symbol}"}

            last_price = float(close_prices.iloc[-1])
            if previous_close is None and len(close_prices) > 1:
                previous_close = float(close_prices.iloc[-2])

        last_price = float(last_price)
        previous_close = float(previous_close) if previous_close is not None else None
        day_high = float(day_high) if day_high is not None else None
        day_low = float(day_low) if day_low is not None else None

        change = round(last_price - previous_close, 2) if previous_close is not None else None
        change_percent = round((change / previous_close) * 100, 2) if previous_close not in (None, 0) and change is not None else None

        return {
            "stock": symbol.upper(),
            "price": round(last_price, 2),
            "previousClose": round(previous_close, 2) if previous_close is not None else None,
            "change": change,
            "changePercent": change_percent,
            "dayHigh": round(day_high, 2) if day_high is not None else None,
            "dayLow": round(day_low, 2) if day_low is not None else None,
            "currency": currency,
            "marketState": market_state,
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
