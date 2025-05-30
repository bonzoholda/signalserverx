from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from collections import deque
import requests
import pandas as pd
import numpy as np
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIG ===
trading_pair = "PI-USDT"
candle_tf = "5m"
fallback_window_sec = 180
fallback_interval_sec = 15
fallback_prices = deque(maxlen=fallback_window_sec // fallback_interval_sec)

# === Signal store ===
latest_signal = {
    "pair": trading_pair,
    "signal": "initializing",
    "price": None,
    "timestamp": None
}

# === Get historical OHLCV from OKX ===
def get_okx_ohlcv(symbol, bar="5m", limit=100):
    try:
        url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={bar}&limit={limit}"
        res = requests.get(url)
        raw = res.json()
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

# === RSI calculation ===
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# === Detect divergence ===
def detect_divergence(df, lookback=20):
    from scipy.signal import argrelextrema
    df = df.copy()

    # Ensure RSI is present
    if 'rsi' not in df.columns:
        df = calculate_rsi(df)

    # Find local minima/maxima in close prices
    df['min_idx'] = df['close'].iloc[argrelextrema(df['close'].values, np.less_equal, order=3)[0]]
    df['max_idx'] = df['close'].iloc[argrelextrema(df['close'].values, np.greater_equal, order=3)[0]]

    # Recent local extrema
    recent_lows = df[df['min_idx'].notnull()][['timestamp', 'close', 'rsi']].tail(lookback)
    recent_highs = df[df['max_idx'].notnull()][['timestamp', 'close', 'rsi']].tail(lookback)

    bullish_div = False
    bearish_div = False

    # Bullish divergence: price makes lower low, RSI makes higher low
    if len(recent_lows) >= 2:
        p1, p2 = recent_lows.iloc[-2], recent_lows.iloc[-1]
        if p2['close'] < p1['close'] and p2['rsi'] > p1['rsi']:
            bullish_div = True

    # Bearish divergence: price makes higher high, RSI makes lower high
    if len(recent_highs) >= 2:
        p1, p2 = recent_highs.iloc[-2], recent_highs.iloc[-1]
        if p2['close'] > p1['close'] and p2['rsi'] < p1['rsi']:
            bearish_div = True

    return bullish_div, bearish_div


# === Detect short trend ===
def detect_short_trend(df):
    df = df.copy()
    df['green'] = df['close'] > df['open']
    df['red'] = df['close'] < df['open']

    df['uptrend'] = (
        (df['green'])
        & (df['close'] > df['close'].shift(1))
        & (df['close'].shift(1) > df['close'].shift(2))
        & (df['green'].shift(1))
        & (df['green'].shift(2))
    )
    
    df['downtrend'] = (
        (df['red'])
        & (df['close'] < df['close'].shift(1))
        & (df['close'].shift(1) < df['close'].shift(2))
        & (df['red'].shift(1))
        & (df['red'].shift(2))
    )


    return df


# === Signal generator ===
def generate_signals(df, local_window=5):
    df['sma'] = df['close'].rolling(window=20).mean()
    deviation = 0.01
    df['upper_band'] = df['sma'] * (1 + deviation)
    df['lower_band'] = df['sma'] * (1 - deviation)
    df = calculate_rsi(df)
    df = detect_short_trend(df)  # <-- Add this here
    
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

    # Reset first
    df['signal'] = 0
    df['signal_type'] = None

    # Apply trend-based signals first
    df.loc[df['uptrend'], 'signal'] = 1
    df.loc[df['uptrend'], 'signal_type'] = 'buy'

    df.loc[df['downtrend'], 'signal'] = -1
    df.loc[df['downtrend'], 'signal_type'] = 'sell'
    
    return df

# === Fallback price fetch from OKX public REST ===
def fetch_current_price(pair):
    try:
        url = f"https://www.okx.com/api/v5/market/ticker?instId={pair}"
        response = requests.get(url)
        data = response.json()
        price_str = data['data'][0].get('last', None)
        if price_str and price_str.strip() != '':
            return float(price_str)
        else:
            return None
    except Exception as e:
        print(f"[FALLBACK ERROR] {type(e).__name__}: {e}")
        return None

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
                price = fetch_current_price(trading_pair)
                if price:
                    fallback_prices.append(price)

                    if len(fallback_prices) >= fallback_prices.maxlen:
                        df = pd.DataFrame(list(fallback_prices), columns=['close'])
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
                            "price": price,
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
                else:
                    latest_signal = {
                        "pair": trading_pair,
                        "signal": "invalid-price",
                        "price": None,
                        "timestamp": int(time.time())
                    }

        except Exception as e:
            print(f"[Loop Error] {type(e).__name__}: {e}")
        time.sleep(fallback_interval_sec)

# === Background task ===
@app.on_event("startup")
def start_signal_loop():
    Thread(target=signal_loop, daemon=True).start()

# === API endpoint ===
@app.get("/api/signal")
def get_signal():
    return JSONResponse(latest_signal)
