import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import seaborn as sns

# Apply Seaborn styling
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Inject custom CSS for background and UI improvements
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://github.com/eduar-infante/capstone-pytorch/blob/main/nick-chong-N__BnvQ_w18-unsplash%20(1).jpg?raw=true");
        background-attachment: fixed;
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }

    /* Dark overlay for better readability */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }

    /* Main content block */
    .block-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        max-width: 850px;
        margin: auto;
    }

    /* Title */
    .stApp h1 {
        color: #FFA500;
        font-size: 2.8em;
        font-weight: 700;
        text-align: center;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 10px;
        padding: 0.8em 1.6em;
        font-size: 1em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #004c99;
    }

    /* Selectbox */
    .stSelectbox > div {
        background-color: #ffffff;
        color: #333333;
        border-radius: 10px;
    }

    /* Slider label */
    .stSlider > div {
        color: #ffffff;
    }

    /* Dataframe container */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to fetch stock data
def fetch_latest_data(stock, years=5):
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=years * 365)).strftime('%Y-%m-%d')
    stock_data = yf.download(stock, start=start_date, end=end_date)
    return stock_data

# Function to create sequences
def create_sequences(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, 0])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

# Function to fine-tune the model
def fine_tune_model(stock):
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.npy"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        st.write(f"🔄 Loading existing model for {stock}...")

        model = load_model(model_path)
        latest_data = fetch_latest_data(stock, years=1)
        X_train, y_train, scaler = create_sequences(latest_data[['Close']].values)

        st.write(f"🔧 Fine-tuning model for {stock}...")
        model.fit(X_train, y_train, epochs=3, batch_size=16, verbose=1)

        model.save(model_path)
        np.save(scaler_path, scaler)
        st.success(f"✅ Model for {stock} fine-tuned and saved successfully!")

    else:
        st.error(f"No saved model found for {stock}. Please train the model first.")

# Function to predict future prices
def predict_future(stock, days=30):
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.npy"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(f"No saved model found for {stock}. Train or fine-tune the model first.")
        return None

    model = load_model(model_path)
    scaler = np.load(scaler_path, allow_pickle=True).item()

    stock_data = fetch_latest_data(stock, years=5)
    last_50_days = stock_data[['Close']].values[-50:]
    last_50_days_scaled = scaler.transform(last_50_days)

    predictions = []
    for _ in range(days):
        X_input = np.array([last_50_days_scaled], dtype=np.float32)
        next_pred = model.predict(X_input)[0, 0]
        predictions.append(next_pred)
        last_50_days_scaled = np.vstack([last_50_days_scaled[1:], [next_pred]])

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Streamlit UI Layout
st.title("📈 Stock Market LSTM Prediction & Fine-Tuning")

with st.container():
    stocks = ["TSLA", "NVDA", "NFLX", "AAL"]
    selected_stock = st.selectbox("Select a stock:", stocks)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Load Latest Data"):
            data = fetch_latest_data(selected_stock, years=1)
            st.write(f"📊 Latest 1-Year Stock Data for {selected_stock}:")
            st.dataframe(data.tail())
            st.line_chart(data["Close"])

    with col2:
        if st.button("🛠️ Fine-Tune Model"):
            fine_tune_model(selected_stock)

    st.markdown("### 🔮 Prediction Settings")
    days_to_predict = st.slider("Days to predict:", min_value=1, max_value=30, value=10)

    if st.button("🚀 Predict Future Prices"):
        future_prices = predict_future(selected_stock, days=days_to_predict)

        if future_prices is not None:
            future_dates = [datetime.datetime.today() + datetime.timedelta(days=i) for i in range(days_to_predict)]
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices.flatten()})

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"#### 📅 Future Stock Prices for {selected_stock}:")
                future_df["Date"] = future_df["Date"].dt.strftime('%Y-%m-%d')
                st.dataframe(future_df)

            with col2:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(future_df["Date"], future_df["Predicted Price"], marker='o', linestyle='dashed', color='#FF5733', label="Predicted")
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Price")
                ax.set_title(f"Predicted Stock Prices for {selected_stock}")
                plt.xticks(rotation=30)
                ax.legend()
                st.pyplot(fig)

            past_data = fetch_latest_data(selected_stock, years=5).reset_index()[["Date", "Close"]]
            past_data.columns = ["Date", "Stock Price"]

            split_idx = int(len(past_data) * 0.8)
            train_data = past_data[:split_idx]
            val_data = past_data[split_idx:]

            train_data["Date"] = pd.to_datetime(train_data["Date"])
            val_data["Date"] = pd.to_datetime(val_data["Date"])
            future_df["Date"] = pd.to_datetime(future_df["Date"])

            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(train_data["Date"], train_data["Stock Price"], color='blue', linewidth=2, label="Train")
            ax2.plot(val_data["Date"], val_data["Stock Price"], color='orange', linewidth=2, label="Validation")
            ax2.plot(future_df["Date"], future_df["Predicted Price"], color='red', linestyle='dashed', linewidth=2, label="Predictions")

            ax2.xaxis.set_major_locator(mdates.YearLocator(1))
            ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.xticks(rotation=30)
            ax2.set_ylabel("Stock Price (USD)")
            ax2.set_title(f"Stock Price Trends for {selected_stock} (Train, Validation & Predictions)")
            ax2.legend()

            st.pyplot(fig2)

            # Plot actual price (last 2 months) + predicted price (next X days)
            # 60 days = 2 months approx
            last_2_months_data = fetch_latest_data(selected_stock, years=1).reset_index()
            last_2_months_data = last_2_months_data[["Date", "Close"]].tail(60)

            # Ensure the dates are datetime for consistent plotting
            last_2_months_data["Date"] = pd.to_datetime(last_2_months_data["Date"])
            future_df["Date"] = pd.to_datetime(future_df["Date"])

            fig3, ax3 = plt.subplots(figsize=(14, 5))

            # Plot actual last 2 months
            ax3.plot(last_2_months_data["Date"], last_2_months_data["Close"],
                    color='blue', linewidth=2, label="Actual Price (Last 2 Months)")

            # Plot future predicted prices
            ax3.plot(future_df["Date"], future_df["Predicted Price"],
                    color='red', linewidth=2, linestyle='dashed', label=f"Predicted Price (Next {days_to_predict} Days)")

            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax3.xaxis.set_major_locator(mdates.DayLocator(interval=7))

            plt.xticks(rotation=45)
            ax3.set_xlabel("Date")
            ax3.set_ylabel("Stock Price (USD)")
            ax3.set_title(f"{selected_stock} - Last 2 Months Actual & Next {days_to_predict} Days Prediction")
            ax3.legend()

            st.pyplot(fig3)
