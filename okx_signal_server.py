from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
import os
import requests
import pandas as pd
import numpy as np
import time
import gc
from dotenv import load_dotenv
from okx.MarketData import MarketAPI


load_dotenv()

app = FastAPI()

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trading_pair = "PI-USDT"

# OKX API client (no need for auth for public market data)
market_api = MarketAPI()

latest_signal = {
    "pair": trading_pair,
    "signal": "initializing",
    "price": None,
    "timestamp": None
}

# Fetch historical OHLCV candles from OKX
def get_okx_ohlcv(symbol=trading_pair, bar="5m", limit=100):
    try:
        # OKX returns candles in reverse-chronological order
        raw = market_api.get_candlesticks(instId=symbol, bar=bar, limit=limit)
        df = pd.DataFrame(raw['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'volumeCcy', 'volumeCcyQuote', 'confirm'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype('float32')

        df.sort_values('timestamp', inplace=True)  # Ensure oldest first
        return df
    except Exception as e:
        print(f"[OKX fetch error] {type(e).__name__}: {e}")
        print(f"[DEBUG] Sample raw data: {raw['data'][0]}")
        print(f"[DEBUG] Columns returned: {len(raw['data'][0])}")

        return pd.DataFrame()

# RSI Calculation
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# Divergence
def detect_divergence(df, lookback=20):
    df = df.copy()
    df['local_min'] = df['close'][df['close'] == df['close'].rolling(window=5, center=True).min()]
    df['local_max'] = df['close'][df['close'] == df['close'].rolling(window=5, center=True).max()]

    recent_lows = df[['timestamp', 'close', 'rsi']][df['local_min'].notnull()].tail(lookback)
    recent_highs = df[['timestamp', 'close', 'rsi']][df['local_max'].notnull()].tail(lookback)

    bullish_div = False
    bearish_div = False

    if len(recent_lows) >= 2:
        p1, p2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        if p2['close'] < p1['close'] and p2['rsi'] > p1['rsi']:
            bullish_div = True

    if len(recent_highs) >= 2:
        p1, p2 = recent_highs.iloc[-2], recent_highs.iloc[-1]
        if p2['close'] > p1['close'] and p2['rsi'] < p1['rsi']:
            bearish_div = True

    return bullish_div, bearish_div

# Signal generation logic
def generate_signals(df, local_window=5):
    df['sma'] = df['close'].rolling(window=20).mean()
    deviation = 0.01
    df['upper_band'] = df['sma'] * (1 + deviation)
    df['lower_band'] = df['sma'] * (1 - deviation)
    df = calculate_rsi(df)

    df['rolling_min'] = df['close'].rolling(window=local_window).min()
    df['rolling_max'] = df['close'].rolling(window=local_window).max()

    tolerance = 1e-8
    buy = (
        np.isclose(df['close'].shift(1), df['rolling_min'].shift(1), atol=tolerance)
        & (df['close'] > df['close'].shift(1))
        & (df['close'] < df['lower_band'])
    )
    sell = (
        np.isclose(df['close'].shift(1), df['rolling_max'].shift(1), atol=tolerance)
        & (df['close'] < df['close'].shift(1))
        & (df['close'] > df['upper_band'])
    )

    df['signal'] = 0
    df['signal_type'] = None
    df.loc[buy, 'signal'] = 1
    df.loc[buy, 'signal_type'] = 'buy'
    df.loc[sell, 'signal'] = -1
    df.loc[sell, 'signal_type'] = 'sell'

    return df

# Signal loop using OKX price feed
def signal_loop():
    global latest_signal
    pair = trading_pair
    while True:
        try:
            df = get_okx_ohlcv(symbol=pair, bar="5m", limit=100)
            if df.empty:
                time.sleep(15)
                continue

            df = generate_signals(df)
            bull_div, bear_div = detect_divergence(df)
            last = df.iloc[-1]
            signal = last['signal_type']

            if bull_div:
                sig = 'long-divergence'
            elif bear_div:
                sig = 'short-divergence'
            elif signal == 'buy':
                sig = 'long'
            elif signal == 'sell':
                sig = 'short'
            else:
                sig = 'no-signals'

            latest_signal = {
                "pair": pair,
                "signal": sig,
                "price": float(last['close']),  # Ensure serializable
                "timestamp": int(time.time())
            }


            print(f"[OKX SIGNAL] {sig} @ {last['close']}")
        except Exception as e:
            print(f"[Loop Error] {type(e).__name__}: {e}")
        time.sleep(15)

@app.on_event("startup")
def start_loop():
    Thread(target=signal_loop, daemon=True).start()

@app.get("/api/signal")
def get_signal():
    return JSONResponse(latest_signal)
