from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from collections import deque
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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG ===
trading_pair = "PI-USDT"   # CHANGE THIS IF NEEDED
candle_tf = "15m"
fallback_window_sec = 180
fallback_interval_sec = 15
fallback_prices = deque(maxlen=fallback_window_sec // fallback_interval_sec)

# === OKX API client (public access) ===
market_api = MarketAPI()

# === Global signal storage ===
latest_signal = {
    "pair": trading_pair,
    "signal": "initializing",
    "price": None,
    "timestamp": None
}

# === Fetch historical OHLCV ===
def get_okx_ohlcv(symbol=trading_pair, bar=candle_tf, limit=100):
    try:
        raw = market_api.get_candlesticks(instId=symbol, bar=bar, limit=limit)
        df = pd.DataFrame(raw['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'volumeCcy', 'volumeCcyQuote', 'confirm'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype('float32')
        df.sort_values('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"[OHLCV ERROR] {type(e).__name__}: {e}")
        return pd.DataFrame()

# === RSI Calculation ===
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# === Detect Divergence ===
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

# === Generate Signals ===
def generate_signals(df, local_window=5):
    df['sma'] = df['close'].rolling(window=20).mean()
    deviation = 0.01
    df['upper_band'] = df['sma'] * (1 + deviation)
    df['lower_band'] = df['sma'] * (1 - deviation)
    df = calculate_rsi(df)
    df['rolling_min'] = df['close'].rolling(window=local_window).min()
    df['rolling_max'] = df['close'].rolling(window=local_window).max()

    buy = (
        np.isclose(df['close'].shift(1), df['rolling_min'].shift(1))
        & (df['close'] > df['close'].shift(1))
        & (df['close'] < df['lower_band'])
    )
    sell = (
        np.isclose(df['close'].shift(1), df['rolling_max'].shift(1))
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

# === Signal Loop ===
def signal_loop():
    global latest_signal
    while True:
        try:
            df = get_okx_ohlcv(symbol=trading_pair, bar=candle_tf, limit=100)
            if not df.empty:
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
                    "pair": trading_pair,
                    "signal": sig,
                    "price": float(last['close']),
                    "timestamp": int(time.time())
                }
                print(f"[OKX SIGNAL] {sig} @ {last['close']} (OHLCV)")
            else:
                # Fallback mode
                tick = market_api.get_ticker(instId=trading_pair)
                price = float(tick['data'][0]['last'])
                fallback_prices.append(price)

                if len(fallback_prices) >= fallback_prices.maxlen:
                    df = pd.DataFrame(fallback_prices, columns=['close'])
                    df = calculate_rsi(df)
                    df = generate_signals(df)
                    signal = df.iloc[-1]['signal_type']

                    if signal == 'buy':
                        sig = 'long'
                    elif signal == 'sell':
                        sig = 'short'
                    else:
                        sig = 'no-signals'

                    latest_signal = {
                        "pair": trading_pair,
                        "signal": sig + " (fallback)",
                        "price": float(price),
                        "timestamp": int(time.time())
                    }
                    print(f"[FALLBACK SIGNAL] {sig} @ {price}")
                else:
                    latest_signal = {
                        "pair": trading_pair,
                        "signal": "gathering-fallback",
                        "price": price,
                        "timestamp": int(time.time())
                    }
                    print(f"[FALLBACK] logging... {price}")
        except Exception as e:
            print(f"[Loop Error] {type(e).__name__}: {e}")
        time.sleep(fallback_interval_sec)

# === Start Background Thread ===
@app.on_event("startup")
def start_loop():
    Thread(target=signal_loop, daemon=True).start()

# === Signal API ===
@app.get("/api/signal")
def get_signal():
    return JSONResponse(latest_signal)
