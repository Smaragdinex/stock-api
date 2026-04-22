from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

@app.get("/stock/{symbol}")
def get_stock_data(symbol: str):
    try:
        # 加上 auto_adjust=True 讓資料格式更單純
        df = yf.download(symbol, period="1mo", interval="1d", auto_adjust=True)
        
        if df.empty:
            return {"error": f"找不到股票代號: {symbol}"}

        # 關鍵修正：確保只取出 Close 這一欄，並轉換為最單純的格式
        # 我們使用 .iloc[:, 0] 確保只取第一層的數值
        if isinstance(df['Close'], pd.DataFrame):
            close_prices = df['Close'].iloc[:, 0]
        else:
            close_prices = df['Close']
        
        # 計算 MA5
        ma5 = close_prices.rolling(window=5).mean()
        
        result = []
        # 為了保險，我們只取最後 10 天
        recent_count = min(len(df), 10)
        
        for i in range(-recent_count, 0):
            try:
                date_str = df.index[i].strftime("%Y-%m-%d")
                # 使用 float() 轉換前先取 .item() 確保它是純數值
                price_val = float(close_prices.iloc[i])
                ma5_val = float(ma5.iloc[i]) if pd.notnull(ma5.iloc[i]) else None
                
                result.append({
                    "date": date_str,
                    "price": round(price_val, 2),
                    "ma5": round(ma5_val, 2) if ma5_val else None
                })
            except Exception:
                continue
                
        return {
            "stock": symbol.upper(),
            "data": result
        }
    except Exception as e:
        return {"error": f"伺服器內部錯誤: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)