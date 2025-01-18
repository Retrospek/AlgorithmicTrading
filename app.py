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
import joblib
import numpy as np


app = Flask(__name__)

@__cached__
def load_models():
    lstm = torch.load(r"backend\ArtificialIntelligence\predictors\model")

    scaler = joblib.load(r"backend\ArtificialIntelligence\scalers\scaler.pkl")

    return lstm, scaler

lstm, scaler = load_models()


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


class LSTMdataset(Dataset):
    def __init__(self, data, sequence_length, output_length):
        self.data = data
        self.data_values = self.data.values
        self.targetIDX = data.columns.get_loc('RSI')
        self.sequence_length = sequence_length
        self.output_length = output_length
        self.sample_length = sequence_length + output_length
        
        self.valid_indices = []
        for i in range(len(self.data) - self.sample_length + 1): 
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            raise IndexError("Index out of bounds")
        
        dayIDX = self.valid_indices[idx]
        
        history = self.data_values[dayIDX:dayIDX + self.sequence_length]
        forecast = self.data_values[dayIDX + self.sequence_length:dayIDX + self.sample_length, self.targetIDX]
        
        if len(history) != self.sequence_length or len(forecast) != self.output_length:
            raise ValueError(f"Inconsistent sequence length at index {idx}")

        history = torch.tensor(history, dtype=torch.float32)
        forecast = torch.tensor(forecast, dtype=torch.float32)

        return history, forecast


@app.route('/')
def home():
    return 'Welcome to the Prediction API!'
    

@app.route('/predict', methods=['POST']) 
def predict():
    today = str(datetime.today()).split()[0] # grabbing todays date
    sixty_days_ago = (datetime.today() - timedelta(days=59)).strftime('%Y-%m-%d') # Grab 59 days prior inclusive len = 60

    predictor = torch.load(r"backend\ArtificialIntelligence\predictors\model")

    soxx_data = yf.download(tickers=["SOOX"], start=sixty_days_ago, end=today)

    soxx_data.reset_index()

    soxx_data['RSI'] = calculate_rsi(soxx_data['Close'], window=14)
    soxx_data['Return'] = ((soxx_data['Close'] - soxx_data['Open']) / soxx_data['Open'])
    soxx_data = soxx_data[['High', 'Low', 'Volume', 'Open', 'Close', 'Return', 'RSI']]






    try:
        data = request.get_json()
        
        features = np.array(data['features']).reshape(1, -1)  

        prediction = model.predict(features)

        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
