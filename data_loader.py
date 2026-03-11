import os
import requests
import pandas as pd
from datetime import datetime
import time


class BinanceDataLoader:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def fetch_ohlcv(self, symbol, interval, start_str, end_str=None):
        """
        Fetch historical OHLCV data from Binance.

        :param symbol: e.g., 'BTCUSDT'
        :param interval: '1d' or '1h'
        :param start_str: Date string 'YYYY-MM-DD'
        :param end_str: Optional date string 'YYYY-MM-DD'
        """
        print(f"Fetching {symbol} ({interval}) from {start_str} to {end_str or 'now'}...")

        start_ts = int(datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000)
        end_ts = (
            int(datetime.strptime(end_str, "%Y-%m-%d").timestamp() * 1000)
            if end_str
            else int(time.time() * 1000)
        )

        all_data = []
        limit = 1000
        current_ts = start_ts

        while current_ts < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ts,
                "endTime": end_ts,
                "limit": limit,
            }

            response = requests.get(self.BASE_URL, params=params)
            if response.status_code != 200:
                print(f"  API error {response.status_code}: {response.text}")
                break

            data = response.json()

            if not data or not isinstance(data, list):
                break

            all_data.extend(data)

            last_ts = data[-1][0]
            if last_ts == current_ts:
                break
            current_ts = last_ts + 1

            print(f"  Fetched {len(all_data)} candles so far...")
            time.sleep(0.1)

        if not all_data:
            print(f"  No data returned for {symbol} {interval}.")
            return pd.DataFrame()

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
        ]

        df = pd.DataFrame(all_data, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        essential_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[essential_cols]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        before = len(df)
        df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
        after = len(df)
        if before != after:
            print(f"  Removed {before - after} duplicate rows.")

        df = df.dropna().reset_index(drop=True)

        file_name = f"{symbol}_{interval}.csv"
        file_path = os.path.join(self.raw_data_dir, file_name)
        df.to_csv(file_path, index=False)
        print(f"  Saved {len(df)} rows to {file_path}\n")
        return df


if __name__ == "__main__":
    loader = BinanceDataLoader()

    # Updated per supervisor requirement: full historical range from 2017
    start_date = "2017-01-01"
    end_date   = "2026-01-01"

    symbols   = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1d", "1h"]

    for symbol in symbols:
        for interval in intervals:
            try:
                df = loader.fetch_ohlcv(symbol, interval, start_date, end_date)
                if not df.empty:
                    print(f"  Preview {symbol} {interval}:")
                    print(df.head(3))
                    print()
            except Exception as e:
                print(f"  ERROR fetching {symbol} {interval}: {e}")