import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("CryptoCurrency Price Predictor App")

# Default stock symbol
stock = "BTC-USD"

# Define date range
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

# User input for stock symbol
stock = st.text_input("Enter the stock here", stock)

# Fetch stock data
bit_coin_data = yf.download(stock, start, end)

# Check if the data is empty
if bit_coin_data.empty:
    st.error(f"Failed to fetch data for {stock}. Please check the symbol and try again.")
else:
    # Load pre-trained model
    model = load_model("coin_model.keras")

    st.subheader("Crypto Data")
    st.write(bit_coin_data)

    # Splitting data
    splitting_len = int(len(bit_coin_data) * 0.9)
    x_test = bit_coin_data[['Close']].iloc[splitting_len:]

    # Ensure x_test is not empty before plotting
    if not x_test.empty:
        st.subheader('Original Close Price')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(bit_coin_data.Close, 'b')
        st.pyplot(fig)

        st.subheader("Test Close Price")
        st.write(x_test)

        # Scaling data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']].values.reshape(-1, 1))

        # Creating test data
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Ensure x_data is correctly shaped before prediction
        if x_data.shape[0] > 0:
            predictions = model.predict(x_data)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)
        else:
            inv_pre, inv_y_test = [], []

        # Creating DataFrame for visualization
        if len(inv_pre) > 0:
            ploting_data = pd.DataFrame(
                {
                    'original_test_data': inv_y_test.reshape(-1),
                    'predictions': inv_pre.reshape(-1)
                },
                index=bit_coin_data.index[splitting_len + 100:]
            )
            st.subheader("Original values vs Predicted values")
            st.write(ploting_data)

            st.subheader('Original Close Price vs Predicted Close price')
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(pd.concat([bit_coin_data.Close[:splitting_len + 100], ploting_data], axis=0))
            ax.legend(["Data - not used", "Original Test data", "Predicted Test data"])
            st.pyplot(fig)

    # Predicting future prices
    st.subheader("Future Price Values")
    last_100 = bit_coin_data[['Close']].tail(100)
    last_100_scaled = scaler.transform(last_100.values.reshape(-1, 1)).reshape(1, -1, 1)


    # Future prediction function
    def predict_future(no_of_days, prev_100):
        future_predictions = []
        prev_100 = np.array(prev_100, dtype=np.float32)

        for _ in range(int(no_of_days)):
            next_day = model.predict(prev_100)[0]  # Get first predicted value
            future_predictions.append(next_day)
            prev_100 = np.append(prev_100[0], next_day).reshape(1, -1, 1)

        return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


    # Get user input for future prediction
    default_days = 10
    no_of_days = int(st.text_input("Enter the number of days to predict: ", default_days))
    future_results = predict_future(no_of_days, last_100_scaled)

    # Plot future predictions
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(pd.DataFrame(future_results), marker='o')
    for i, val in enumerate(future_results):
        ax.text(i, val, f'{int(val[0])}', fontsize=12)
    ax.set_xlabel('Days')
    ax.set_ylabel('Close Price')
    ax.set_xticks(range(no_of_days))
    ax.set_title(f'Predicted Closing Price for {stock}')
    st.pyplot(fig)
