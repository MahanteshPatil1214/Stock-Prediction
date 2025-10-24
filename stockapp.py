import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction App")
st.markdown("Predict next-day stock prices using Machine Learning + Technical Indicators.")


ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY.NS):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("today"))


@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

if st.button("Load Data"):
    data = load_data(ticker, start_date, end_date)
    st.subheader(f"Raw Data for {ticker}")
    st.dataframe(data.tail())


    data["RSI"] = RSIIndicator(data["Close"].squeeze(), window=14).rsi()
    macd = MACD(data["Close"].squeeze())
    data["MACD"] = macd.macd()
    data["Signal_Line"] = macd.macd_signal()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()


    st.subheader("📉 Stock Closing Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["Close"], label="Close Price")
    ax.plot(data["MA20"], label="MA20", linestyle="--")
    ax.plot(data["MA50"], label="MA50", linestyle="--")
    ax.legend()
    st.pyplot(fig)


    data["Tomorrow"] = data["Close"].shift(-1)
    data = data.dropna()

    features = ["Open", "High", "Low", "Close", "Volume", "RSI", "MACD", "Signal_Line", "MA20", "MA50"]
    X = data[features]
    y = data["Tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    mse = mean_squared_error(y_test, predictions)
    st.subheader("📈 Model Performance")
    st.write(f"Mean Squared Error: **{mse:.4f}**")


    st.subheader("🔮 Actual vs Predicted Prices")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(y_test.values, label="Actual Price", color="blue")
    ax2.plot(predictions, label="Predicted Price", color="red")
    ax2.legend()
    st.pyplot(fig2)


    last_row = X.iloc[-1].values.reshape(1, -1)
    next_day_pred = model.predict(last_row)[0]
    st.success(f"📅 Predicted Closing Price for Next Day: **${next_day_pred:.2f}**")

    st.markdown("---")
    st.info("Note: This model uses basic regression and may not capture market volatility accurately. Use results for educational purposes only.")
