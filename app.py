from flask import Flask, request, jsonify
import pickle
from datetime import datetime, timedelta
import yfinance as yf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import joblib
import schedule
import time

import pandas as pd
import numpy as np

from backend.ArtificialIntelligence.ml import LSTMRegression, LSTMdataset


app = Flask(__name__)

def load_models():
    lstm = torch.load(r"backend\ArtificialIntelligence\predictors\model")
    scaler = joblib.load(r"backend\ArtificialIntelligence\scalers\scaler.pkl")
    return lstm, scaler

lstm, scaler = load_models()
lstm = lstm.to(device)

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict():
    today = str(datetime.today()).split()[0]
    sixty_days_ago = (datetime.today() - timedelta(days=73)).strftime('%Y-%m-%d')

    soxx_data = yf.download(tickers=["SOXX"], start=sixty_days_ago, end=today)
    soxx_data.reset_index()

    soxx_data['RSI'] = calculate_rsi(soxx_data['Close'], window=14)
    soxx_data['Return'] = ((soxx_data['Close'] - soxx_data['Open']) / soxx_data['Open'])
    soxx_data = soxx_data[['High', 'Low', 'Volume', 'Open', 'Close', 'Return', 'RSI']]

    # Drop rows where RSI is NaN (First 14 rows will have NaN RSI)
    print(soxx_data)
    print(soxx_data.columns)
    soxx_data = soxx_data.dropna()

    if len(soxx_data) < 60:
        return jsonify({'error': 'Not enough valid data for RSI'})

    soxx_data_scaled = scaler.transform(soxx_data[['High', 'Low', 'Volume', 'Open', 'Close', 'Return', 'RSI']])

    soxx_data_scaled = pd.DataFrame(soxx_data_scaled, columns=soxx_data.columns)
    
    soxx_data_scaled = soxx_data_scaled.iloc[-60:]

    lstm_dataset = LSTMdataset(soxx_data_scaled, sequence_length=60, output_length=5)
    lstm_dataloader = DataLoader(lstm_dataset, batch_size=1)
    
    predictions = []
    with torch.no_grad():
        for x_batch, y_batch in lstm_dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            print(x_batch.shape)
            prediction = np.squeeze(lstm(x_batch).cpu().numpy())
            predictions.append(prediction)

    rsi_scale = scaler.scale_[6]
    rsi_mean = scaler.mean_[6]
    unscaled_predictions = (np.array(predictions).flatten() * rsi_scale) + rsi_mean

    print(unscaled_predictions)

    return jsonify({'prediction': unscaled_predictions.tolist()})

def run_daily_prediction():

    with app.app_context():  
        predict()

# Schedule the task to run every day at a set time, e.g., 8:00 AM
# schedule.every().day.at("08:00").do(run_daily_prediction)

if __name__ == '__main__':
    with app.app_context():  # Make sure we are within the application context
        print("Starting prediction...")
        predict()

    # Uncomment the following to enable daily scheduling
    # print("Starting scheduler...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)  # Wait for a second before checking again
