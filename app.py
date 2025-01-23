import io
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

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
import matplotlib.pyplot as plt

from backend.ArtificialIntelligence.ml import LSTMRegression, LSTMdataset

app = Flask(__name__)
CORS(app)

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
    return render_template(r"home.html")

@app.route("/predict", methods=["GET"])
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

    if len(soxx_data) < 60:
        return jsonify({'error': 'Not enough data for prediction'})

    # Scale data
    soxx_data_scaled = scaler.transform(soxx_data)
    soxx_data_scaled = pd.DataFrame(soxx_data_scaled, columns=soxx_data.columns)
    soxx_data_scaled = soxx_data_scaled.iloc[-60:] # Last 60 days worth of data and then it outputs the next 5 days

    # Create LSTM dataset
    lstm_dataset = LSTMdataset(soxx_data_scaled, sequence_length=60)
    print(f"Length of LSTM Dataset: {len(lstm_dataset)}")
    lstm_dataloader = DataLoader(lstm_dataset, batch_size=1)

    predictions = []
    with torch.no_grad():
        for x_batch in lstm_dataloader:
            x_batch = x_batch.to(device)
            prediction = lstm(x_batch).cpu().numpy().squeeze()
            predictions.append(prediction)

    rsi_scale = scaler.scale_[6]
    rsi_mean = scaler.mean_[6]
    unscaled_predictions = (np.array(predictions).flatten() * rsi_scale) + rsi_mean

    return jsonify({'prediction': unscaled_predictions.tolist()})

@app.route('/plot', methods=['POST'])
def plot():
    prediction_data = request.json.get('prediction')  
    
    if not prediction_data:
        return jsonify({'error': 'No prediction data provided'}), 400
    
    plt.figure(figsize=(10, 6), dpi=100)
    
    plt.style.use('seaborn')
    plt.plot(range(1, len(prediction_data) + 1), prediction_data, 
            marker='o', 
            linestyle='-', 
            linewidth=2, 
            color='#2196F3')
    
    plt.title('RSI Prediction Trend Analysis', fontsize=14, pad=20)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('RSI Value', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axhline(y=70, color='#FF5252', linestyle='--', alpha=0.5, label='Overbought (70)')
    plt.axhline(y=30, color='#4CAF50', linestyle='--', alpha=0.5, label='Oversold (30)')
    
    plt.legend(['Predicted RSI', 'Overbought Level', 'Oversold Level'], 
              loc='best', 
              frameon=True)
    
    plt.tight_layout()
    
    # Save plot to a BytesIO object with higher quality
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    
    return send_file(img, mimetype='image/png')



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

