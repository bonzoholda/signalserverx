import pandas as pd
import numpy as np
import requests
import time
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

os.makedirs("models", exist_ok=True)



class MLSignalGeneratorOKX:
    def __init__(self, symbol="PI-USDT", interval="15m", train=False):
        self.symbol = symbol
        self.interval = interval

        self.model_path = f"models/ml_model_{symbol.replace('-', '')}.pkl"

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        if train:
            self.train_model()
        else:
            try:
                self.model = joblib.load(self.model_path)
            except:
                print("Model not found. Training...")
                self.train_model()

    def fetch_ohlcv(self, limit=1000):
        url = f"https://www.okx.com/api/v5/market/candles?instId={self.symbol}&bar={self.interval}&limit={limit}"
        resp = requests.get(url)
        data = resp.json()
        if data["code"] != "0":
            raise Exception(f"OKX error: {data['msg']}")
        
        df = pd.DataFrame(data["data"], columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "volCcy", "volCcyQuote", "confirm"
        ])

        df = df.iloc[::-1]  # Oldest first
        df = df.astype({
            "open": float, "high": float, "low": float, 
            "close": float, "volume": float
        })

        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="ms")

        
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def add_indicators(self, df):
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df.dropna(inplace=True)
        return df

    def create_labels(self, df, threshold=0.002):
        df['future_return'] = df['close'].pct_change(periods=5).shift(-5)
        df['signal'] = 0
        df.loc[df['future_return'] > threshold, 'signal'] = 1
        df.loc[df['future_return'] < -threshold, 'signal'] = -1
        df.dropna(inplace=True)
        return df

    def train_model(self):
        df = self.fetch_ohlcv(limit=300)
        df = self.add_indicators(df)
        df = self.create_labels(df)
        X = df[['rsi', 'sma', 'macd', 'macd_signal']]
        y = df['signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        print(f"Trained model saved as {self.model_path}")
        print(f"Accuracy: {self.model.score(X_test, y_test):.2%}")

    def predict_signal(self):
        df = self.fetch_ohlcv(limit=100)
        df = self.add_indicators(df)
        latest = df.tail(1)
        X_live = latest[['rsi', 'sma', 'macd', 'macd_signal']]
        pred = self.model.predict(X_live)[0]
        return {1: "BUY", -1: "SELL", 0: "HOLD"}[pred]
