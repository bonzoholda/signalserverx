from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from collections import deque
import requests
import pandas as pd
import numpy as np
import time
from ml_signal_generator_okx import MLSignalGeneratorOKX


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
candle_tf = "15m"
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

generator = MLSignalGeneratorOKX(symbol=trading_pair, interval=candle_tf)
signal = generator.predict_signal()
print("ML Signal:", signal)


# === Get historical OHLCV from OKX ===
def get_okx_ohlcv(symbol, bar="15m", limit=100):
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

# === 3 candles volume confirmation ===
def ta_3_candle_signal(df):
    if len(df) < 3:
        return "HOLD"

    # Last 3 closes
    c1, c2, c3 = df['close'].iloc[-3:]
    v1, v2, v3 = df['volume'].iloc[-3:]
    
    # Bullish pattern: 3 rising closes
    if c1 < c2 < c3 and v1 < v2 < v3:
        return "long"

    # Bearish pattern: 3 falling closes
    if c1 > c2 > c3 and v1 > v2 > v3:
        return "short"

    return "HOLD"



# === Measure market strength ===
def get_market_strength(df):
    """
    Analyzes RSI to determine market strength.
    Returns: 'strong_trend', 'choppy', or 'neutral'
    """
    df = calculate_rsi(df)

    rsi = df.iloc[-1]['rsi']

    if rsi >= 65 or rsi <= 35:
        return 'strong_trend'
    elif 45 <= rsi <= 55:
        return 'choppy'
    else:
        return 'neutral'


# === Get DCA trigger multiplier ===
def get_dca_trigger_price(entry_price, stop_loss, market_strength, signal_side):
    """
    Calculates DCA trigger price based on SL and market strength.
    For long → trigger below entry
    For short → trigger above entry
    """
    if market_strength == "choppy":
        multiplier = 5
    elif market_strength == "strong_trend":
        multiplier = 3
    else:
        multiplier = 4

    if signal_side == 'long':
        return round(entry_price - (stop_loss * multiplier), 5)
    elif signal_side == 'short':
        return round(entry_price + (stop_loss * multiplier), 5)
    else:
        return None  # No valid signal




# === Dynamic TP/SL calculation ===
def get_dynamic_tp_sl(df, risk_reward_ratio=1.5, atr_period=14):
    """
    Calculates TP and SL based on recent volatility (using ATR).
    """
    df = df.copy()
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift())
    df['low_close'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()

    last_close = df.iloc[-1]['close']
    last_atr = df.iloc[-1]['atr']

    if pd.isna(last_atr):
        return None, None  # Not enough data yet

    stop_loss = last_atr  # 1x ATR as SL
    take_profit = stop_loss * risk_reward_ratio  # TP = RR * SL

    return round(take_profit, 5), round(stop_loss, 5)


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

                # Traditional signal
                ta_signal = last['signal_type']

                # ML signal
                ml_signal = generator.predict_signal()
                ml_signal = ml_signal.strip().lower() if ml_signal else None
                print(f"[ML] Predicted signal: {ml_signal}")

                # === Priority signal logic (ML first for testing) ===
                if ml_signal == 'buy':
                    sig = 'long'
                elif ml_signal == 'sell':
                    sig = 'short'
                    print(f"[ML SELECTED] Final signal: {sig}")
                elif ml_signal == "hold":
                    ta_3 = ta_3_candle_signal(df)
                    if ta_3 in ["long", "short"]:
                        sig = ta_3
                elif bull_div:
                    sig = 'long-hold'
                elif bear_div:
                    sig = 'short-hold'
                elif ta_signal == 'buy':
                    sig = 'long-hold'
                elif ta_signal == 'sell':
                    sig = 'short-hold'
                else:
                    sig = 'no-signals'

                # === Dynamic TP, SL, and DCA trigger calculation ===
                tp, sl = get_dynamic_tp_sl(df)
                market_strength = get_market_strength(df)
                dca_trigger_price = get_dca_trigger_price(
                    entry_price=last['close'],
                    stop_loss=sl,
                    market_strength=market_strength,
                    signal_side=sig if sig in ['long', 'short'] else None
                )


                latest_signal = {
                    "pair": trading_pair,
                    "signal": sig,
                    "price": float(last['close']),
                    "tp": tp,
                    "sl": sl,
                    "dca_trigger": dca_trigger_price,
                    "market_strength": market_strength,
                    "timestamp": int(time.time())
                }

                print(f"[OKX SIGNAL] Final: {sig} @ {last['close']} (OHLCV + ML)")
                print(f"[DCA] Market: {market_strength}, Trigger: {dca_trigger_price}")

            else:
                # === Fallback mode ===
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
                        print(f"[FALLBACK] Logging... {price}")
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
