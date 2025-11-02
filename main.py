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
# Also use the 9, 21, 50, and 100 EMAs.
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

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss

    # calculate RSI column
    df["rsi"] = 100 - (100 / (1 + rs))

    # ensure df is chronologically sorted before ewm()
    df = df.sort_values("timestamp")

    # calculate EMAs recursively (adjust=False)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    # signal line is the 9-day EMA of the MACD itself
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    print(f"\nDataframe for {ticker}: ")
    print("\n", df.head(10), "\n")

    return df


if __name__ == "__main__":
    ticker = input("\nEnter ticker: ")

    # grab dates, (probably optional) fallback for leap years
    end_date = date.today()
    try:
        start_date = end_date.replace(year=end_date.year - 5)
    except:
        start_date = end_date.replace(month=2, day=28, year=end_date.year - 5)

    df = get_stock_data(ticker, start_date, end_date)

    # fig = plot_stock_data(df, ticker)
    # fig.show(config={"scrollZoom": True})
