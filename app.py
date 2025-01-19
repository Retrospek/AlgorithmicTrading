from flask import Flask, request, jsonify
import pickle
from datetime import datetime, timedelta
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import joblib
import schedule
import time

import pandas as pd
import numpy as np

from backend.ArtificialIntelligence.ml import LSTMRegression, LSTMdataset

app = Flask(__name__)

def load_models():
    """Load LSTM model and scaler from files."""
    lstm = torch.load(r"backend\ArtificialIntelligence\predictors\model")
    scaler = joblib.load(r"backend\ArtificialIntelligence\scalers\scaler.pkl")
    return lstm, scaler

lstm, scaler = load_models()
lstm = lstm.to(device) # Move to GPU

def calculate_rsi(data, window=14):
    """Calculate the RSI (Relative Strength Index) for a data series."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@app.route('/')
@app.route('/home')
def home():
    "Home page initialization"

    

@app.route("/predict", methods=["POST"])
def predict():
    """
    Perform LSTM predictions for the given stock data.
    Utilize standard scalar for inputs, and then unscale outputs
    """
    today = datetime.today()
    start_date = (today - timedelta(days=200)).strftime('%Y-%m-%d')

    soxx_data = yf.download(tickers=["SOXX"], start=start_date, end=today)
    soxx_data.reset_index()

    soxx_data['RSI'] = calculate_rsi(soxx_data['Close'], window=14)
    soxx_data['Return'] = ((soxx_data['Close'] - soxx_data['Open']) / soxx_data['Open'])
    soxx_data = soxx_data[['High', 'Low', 'Volume', 'Open', 'Close', 'Return', 'RSI']]
    soxx_data = soxx_data.dropna()  # Drop rows with NaN values

    if len(soxx_data) < 65:
        return jsonify({'error': 'Not enough data for prediction'})

    # Scale data
    soxx_data_scaled = scaler.transform(soxx_data)
    soxx_data_scaled = pd.DataFrame(soxx_data_scaled, columns=soxx_data.columns)
    soxx_data_scaled = soxx_data_scaled.iloc[-65:]

    # Create LSTM dataset
    lstm_dataset = LSTMdataset(soxx_data_scaled, sequence_length=60, output_length=5)
    lstm_dataloader = DataLoader(lstm_dataset, batch_size=1)

    predictions = []
    with torch.no_grad():
        for x_batch, _ in lstm_dataloader:
            x_batch = x_batch.to(device)
            prediction = lstm(x_batch).cpu().numpy().squeeze()
            predictions.append(prediction)

    rsi_scale = scaler.scale_[6]
    rsi_mean = scaler.mean_[6]
    unscaled_predictions = (np.array(predictions).flatten() * rsi_scale) + rsi_mean

    return jsonify({'prediction': unscaled_predictions.tolist()})

def daily_prediction():
    """Perform daily prediction and save or log results."""
    with app.app_context():
        result = predict()
        print("Daily Prediction: ", result.get_json())

@app.route("/schedule", methods=["POST"])
def schedule_daily_prediction():
    """Start the scheduler to run daily predictions."""
    time_to_run = request.json.get("time", "08:00")  # Default time is 8:00 AM
    schedule.every().day.at(time_to_run).do(daily_prediction)
    return jsonify({"message": f"Daily prediction scheduled at {time_to_run}."})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)

    # Uncomment to enable daily scheduling
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)  # Check schedule every second
