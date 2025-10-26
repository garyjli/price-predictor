import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import date
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data import TimeFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
# Combine all of this with price action and other indicators (will later be determined and tested).


load_dotenv()


# Plots price, volume, and MACD as three separate charts
def plot_stock_data(df, symbol="NVDA"):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.1,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
        ],
        subplot_titles=("Candlesticks with SMAs", "Volume", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    for sma, color in [("SMA5", "blue"), ("SMA20", "orange"), ("SMA50", "green")]:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[sma], name=sma, line=dict(color=color, width=1)
            ),
            row=1,
            col=1,
        )

    volume_colors = [
        "green" if close > open else "red"
        for close, open in zip(df["close"], df["open"])
    ]

    # Check if volume data exists and has valid values
    if "volume" in df.columns and not df["volume"].isna().all():
        # Always display volume in millions
        volume_data = df["volume"] / 1_000_000
        volume_suffix = "M"
        print(f"Max volume: {df['volume'].max():,}")  # Debug print

        print(
            f"Volume data range: {volume_data.min():.1f} to {volume_data.max():.1f} {volume_suffix}"
        )

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=volume_data,
                name="Volume",
                marker_color=volume_colors,
                showlegend=False,
                opacity=0.7,
            ),
            row=2,
            col=1,
        )

        # Set y-axis title for volume
        fig.update_yaxes(title_text=f"Volume ({volume_suffix})", row=2, col=1)
        fig.update_yaxes(tickformat=".1f", ticksuffix=volume_suffix, row=2, col=1)
    else:
        print("Warning: No volume data available or all volume values are NaN")

    if "MACD" in df.columns and not df["MACD"].isna().all():
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=1)),
            row=3,
            col=1,
        )

    if "MACD_Signal" in df.columns and not df["MACD_Signal"].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                line=dict(width=1, dash="dash"),
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
        title=f"{symbol} CHART",
        height=800,
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_white",
        dragmode="pan",
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        bargap=0.05,  # Reduce gap to make candlesticks wider
    )

    return fig


def prepare_data(df, look_back=10):
    # FIXED: Use correct split ratios for NVDA
    splits = [
        ("2021-07-20", 4),  # NVDA 4-for-1 split on July 20, 2021
        ("2024-06-10", 10, True),  # NVDA 10-for-1 split on June 10, 2024 (intraday)
    ]
    df = adjust_for_stock_splits(df, splits)

    print(f"Volume data after split adjustment:")
    print(f"Min volume: {df['volume'].min():,}")
    print(f"Max volume: {df['volume'].max():,}")
    print(f"Volume data type: {df['volume'].dtype}")
    print(f"Any NaN values in volume: {df['volume'].isna().any()}")

    df["SMA5"] = df["close"].rolling(window=5).mean()
    df["SMA20"] = df["close"].rolling(window=20).mean()
    df["SMA50"] = df["close"].rolling(window=50).mean()

    df["Price_Change"] = df["close"].diff()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df_clean = df.dropna().copy()

    return df_clean


def adjust_for_stock_splits(df, splits):
    """
    Adjust historical data for stock splits
    Args:
        df: DataFrame with OHLCV data
        splits: List of tuples (date, ratio) or (date, ratio, is_intraday)
                where ratio is the split ratio (e.g., 4 for a 4:1 split)
    """
    df = df.copy()

    # Debug: Check original volume data
    print(f"Original volume data:")
    print(f"Min volume: {df['volume'].min():,}")
    print(f"Max volume: {df['volume'].max():,}")

    idx = df.index.tz_convert("UTC").normalize()

    # Sort splits by date ascending (earliest first) to apply adjustments correctly
    splits_sorted = sorted(splits, key=lambda s: s[0])

    # Keep track of cumulative factor
    cumulative_factor = 1.0

    for split_info in splits_sorted:
        date_str, ratio = split_info[0], split_info[1]
        is_intraday = len(split_info) == 3 and split_info[2]
        split_dt = pd.to_datetime(date_str).tz_localize("UTC").normalize()

        print(f"Processing split: {date_str}, ratio: {ratio}, intraday: {is_intraday}")

        # Update cumulative factor
        cumulative_factor *= ratio

        # Apply adjustment to all data before this split
        pre_split_mask = idx < split_dt

        if is_intraday:
            # For intraday splits, handle the split day specially
            day_mask = idx == split_dt
            if day_mask.any():
                # Get the post-split close price and work backwards
                post_close = df.loc[day_mask, "close"].iloc[0]
                # Create reasonable OHLC for the split day
                df.loc[day_mask, "open"] = post_close * 0.99
                df.loc[day_mask, "high"] = post_close * 1.02
                df.loc[day_mask, "low"] = post_close * 0.97

        # Adjust all pre-split data
        if pre_split_mask.any():
            print(f"Adjusting {pre_split_mask.sum()} rows for split ratio {ratio}")
            for col in ["open", "high", "low", "close"]:
                df.loc[pre_split_mask, col] /= ratio
            df.loc[pre_split_mask, "volume"] *= ratio

    return df


def create_model(input_shape):
    from tensorflow.keras.regularizers import l2

    model = Sequential(
        [
            LSTM(
                32,
                input_shape=input_shape,
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
    print(f"\nDataframe for {ticker}: ")
    print("\n", df.head(), "\n")

    return df


if __name__ == "__main__":
    ticker = input("Enter ticker: ")

    # grab dates, (probably optional) fallback for leap years
    end_date = date.today()
    try:
        start_date = end_date.replace(year=end_date.year - 5)
    except:
        start_date = end_date.replace(month=2, day=28, year=end_date.year - 5)

    df = get_stock_data(ticker, start_date, end_date)

    # df = prepare_data(raw_df)
    # print("")
    # print(df.head())
    # print("")
    # print("\nDataFrame info:")
    # print(df.info())

    # fig = plot_stock_data(df, ticker)
    # fig.show(config={"scrollZoom": True})
