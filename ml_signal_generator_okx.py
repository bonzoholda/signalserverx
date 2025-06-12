import pandas as pd
import numpy as np
import requests
import time
import joblib
import os
import threading
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

os.makedirs("models", exist_ok=True)

class MLSignalGeneratorOKX:
    def __init__(self, symbol="PI-USDT", interval="15m", train=False, auto_retrain=True):
        self.symbol = symbol
        self.interval = interval
        self.model_path = f"models/ml_model_{symbol.replace('-', '')}.pkl"
        self.data_path = f"models/training_data_{symbol.replace('-', '')}.pkl"
        self.model = RandomForestClassifier(random_state=42)
        self.lock = threading.Lock()

        if train:
            self.train_model()
        else:
            try:
                self.model = joblib.load(self.model_path)
            except:
                print("Model not found. Training...")
                self.train_model()

        if auto_retrain:
            retrain_thread = threading.Thread(target=self.retrain_loop, daemon=True)
            retrain_thread.start()

    def fetch_ohlcv(self, limit=100):
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
        try:
            # Fetch and process new data
            new_df = self.fetch_ohlcv(limit=100)
            new_df = self.add_indicators(new_df)
            new_df = self.create_labels(new_df)

            new_X = new_df[['rsi', 'sma', 'macd', 'macd_signal']]
            new_y = new_df['signal']

            # Load previous data if exists
            if os.path.exists(self.data_path):
                old_X, old_y = joblib.load(self.data_path)
                X = pd.concat([old_X, new_X]).tail(1000)
                y = pd.concat([old_y, new_y]).tail(1000)
            else:
                X, y = new_X, new_y

            # Save combined data for future retrain
            joblib.dump((X, y), self.data_path)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 10],
                'min_samples_split': [2, 5],
            }

            grid = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(X_train, y_train)

            with self.lock:
                self.model = grid.best_estimator_
                joblib.dump(self.model, self.model_path)

            print(f"[{self.symbol}] Model retrained and saved.")
            print(f"[{self.symbol}] Best params: {grid.best_params_}")
            print(f"[{self.symbol}] Accuracy: {self.model.score(X_test, y_test):.2%}")
        except Exception as e:
            print(f"Retraining error: {e}")

    def retrain_loop(self):
        while True:
            print(f"[{self.symbol}] Auto-retraining started.")
            self.train_model()
            print(f"[{self.symbol}] Next retrain in 6h.")
            time.sleep(21600)  # every 6 hours

    def predict_signal(self):
        df = self.fetch_ohlcv(limit=100)
        df = self.add_indicators(df)
        latest = df.tail(1)
        X_live = latest[['rsi', 'sma', 'macd', 'macd_signal']]
        with self.lock:
            pred = self.model.predict(X_live)[0]
        return {1: "BUY", -1: "SELL", 0: "HOLD"}[pred]
