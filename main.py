import os
from datetime import date

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame


# MACD Strategy:
#
# When MACD line crosses ABOVE the signal line (while BELOW the zero line), buy.
# When MACD line crosses BELOW the signal line (while ABOVE the zero line), sell/short.
#
# Use this alongside the 200 EMA.
# If price is above the 200 EMA, then we are in an uptrend.
# If price is below the 200 EMA, then we are in a downtrend.
# We only want to buy during an uptrend and sell during a downtrend.
#
# Also use the 9, 21, 50, and 100 EMAs.
#
# Combine all of this with price action and other indicators (will later be determined and tested).


load_dotenv()


# Fetch open, high, low, close, and volume data for 'ticker' between 'start_date' and 'end_date'
# Returns a dataframe with those as series/columns
def get_stock_data(ticker, start_date, end_date):
    client = StockHistoricalDataClient(
        api_key=os.getenv("APCA_API_KEY_ID"),
        secret_key=os.getenv("APCA_API_SECRET_KEY"),
    )

    # creating request object
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        start=start_date,
        end=end_date,
        timeframe=TimeFrame.Day,
    )

    # due to the structure of the BarSet object returned by get_stock_bars(), performing .df will return a MULTI-INDEX dataframe
    # thus we must convert the current indices into columns with reset_index()
    df = client.get_stock_bars(request).df
    df = df.reset_index()

    # calculate SMAs
    df["sma5"] = df["close"].rolling(window=5).mean()
    df["sma20"] = df["close"].rolling(window=20).mean()
    df["sma50"] = df["close"].rolling(window=50).mean()

    # calculate price change based on close
    df["price_change"] = df["close"].diff()

    # series of price differences
    delta = df["close"].diff()
    # series of positive price differences (any difference < 0 is simply set to 0, not removed)
    gain = delta.clip(lower=0)
    # series of negative price difference MAGNITUDES (any difference > 0 is simply set to 0, not removed)
    loss = -delta.clip(upper=0)
    # 14 days for RSI formula
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ensure df is chronologically sorted before ewm()
    df = df.sort_values("timestamp")
    # calculate EMAs recursively (adjust=False)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    # signal line is the 9-day EMA of the MACD itself
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()

    print(f"\nDataframe for {ticker}: ")
    print("\n", df.head(10))

    return df


# def adjust_for_stock_splits(df, split_date, split_ratio):
#     if isinstance(split_date, str):
#         split_date = pd.to_datetime(split_date).tz_localize("UTC")

#     pre_split_mask = df["timestamp"] < split_date

#     if pre_split_mask.any():
#         price_columns = ["open", "high", "low", "close"]

#         for col in price_columns:
#             if col in df.columns:
#                 df.loc[pre_split_mask, col] = df.loc[pre_split_mask, col] / split_ratio

#         if "volume" in df.columns:
#             df.loc[pre_split_mask, "volume"] = (
#                 df.loc[pre_split_mask, "volume"] * split_ratio
#             )

#     return df


def preprocess_data(df):
    # axis = 1 to specify we want to drop columns
    df = df.drop(
        [
            "symbol",
            "timestamp",
            "open",
            "high",
            "low",
            "vwap",
            "trade_count",
            "ema9",
            "ema21",
            "ema100",
            "sma5",
            "sma20",
            "sma50",
            "rsi",
            "price_change",
        ],
        axis=1,
    )
    # drop any rows with NaN, reset indices, and don't keep old indices as column
    df = df.dropna().reset_index(drop=True)

    # calls astype(int) on a series of booleans
    # direction on day n tells whether price goes up (1) or down (0) from day n to day n+1
    df["direction"] = (df["close"].shift(-1) > df["close"]).astype(int)

    X = df.drop("direction", axis=1)
    Y = df["direction"]

    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    # X_scaled is still a df
    X_scaled = scaler.fit_transform(X)

    window = 60
    X_seq, Y_seq = [], []

    for i in range(len(X_scaled) - window):
        # grabs rows i to i + window from df X_scaled
        X_seq.append(X_scaled[i : i + window])
        # grabs the direction value at index i + window
        Y_seq.append(Y.values[i + window])

    # Now X_seq is a list of python 2D arrays (rows x columns from the df)
    # Now Y_seq is a list of scalars (0 and 1)

    # X_seq now has shape (samples, timesteps, features)
    X_seq = np.array(X_seq)
    # Y_seq now has shape (samples,)
    Y_seq = np.array(Y_seq)

    # # Use 80/20 train/test split
    # split = int(len(X_seq) * 0.8)
    # X_train = X_seq[:split]
    # Y_train = Y_seq[:split]
    # X_test = X_seq[split:]
    # Y_test = Y_seq[split:]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_seq, Y_seq, test_size=0.3, shuffle=False
    )

    print(f"\nPreprocessing for {ticker}: ")
    print("\n", df.head(10), "\n")

    return X_train, Y_train, X_test, Y_test


def create_model(input_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                32,
                return_sequences=True,
                recurrent_dropout=0.2,
                dropout=0.2,
                kernel_regularizer=l2(0.001),
            ),
            LSTM(
                16,
                return_sequences=False,
                recurrent_dropout=0.2,
                dropout=0.2,
                kernel_regularizer=l2(0.001),
            ),
            Dense(8, activation="relu", kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    ticker = input("\nEnter ticker: ")

    # grab dates, (probably optional) fallback for leap years
    end_date = date.today()
    try:
        start_date = end_date.replace(year=end_date.year - 5)
    except:
        start_date = end_date.replace(month=2, day=28, year=end_date.year - 5)

    # fetch data
    df = get_stock_data(ticker, start_date, end_date)

    # df = adjust_for_stock_splits(df, split_date="2021-07-20", split_ratio=4)
    # df = adjust_for_stock_splits(df, split_date="2024-06-10", split_ratio=10)

    # preprocess data
    X_train, Y_train, X_test, Y_test = preprocess_data(df.copy())

    # create model
    model = create_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(
        X_train,
        Y_train,
        epochs=40,
        batch_size=32,
        validation_data=(X_test, Y_test),
        shuffle=False,
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy:.3f}")
