from fastapi import FastAPI
import yfinance as yf
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/stock/{symbol}")
def get_stock_data(symbol: str, period: str = "6mo"):
    try:
        allowed_periods = ["1mo", "3mo", "6mo", "1y"]
        if period not in allowed_periods:
            return {"error": f"不支援的 period: {period}"}

        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True)

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
                date_str = df.index[i].strftime("%Y-%m-%d")
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
            "data": result
        }

    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}