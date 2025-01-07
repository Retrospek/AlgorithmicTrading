import pandas as pd
pd.set_option('display.width', 1000)

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import nasdaqdatalink

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    VotingRegressor, StackingRegressor, BaggingRegressor
)
import xgboost as xgb
from xgboost import XGBRegressor

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load():
    NVDA = 'NVDA'
    SEMICONDUCTORS = ["TSM", "AVGO", "AMD", "ASML", "MRVL", "ON", "NVDA"]
    scaler = MinMaxScaler(feature_range=(0, 1)) # Scaling RSI values for more relatable trends

    semiConductor_data = yf.download(NVDA, start='2010-01-01', end='2024-01-01')
    semiConductor_data.columns = [col[0] for col in semiConductor_data.columns]
    semiConductor_data['Ticker'] = NVDA
    semiConductor_data['RSI'] = calculate_rsi(semiConductor_data['Close'], window=14)
    semiConductor_data['RSI_Scaled'] = scaler.fit_transform(semiConductor_data['RSI'].values.reshape(-1, 1))
    semiConductor_data.reset_index(inplace=True)
    semiConductor_data = semiConductor_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker', 'RSI', 'RSI_Scaled']]
                                                                
    #print(f"Total Data Length: {len(semiConductor_data)}")

    for stock in SEMICONDUCTORS:
        stock_data = yf.download(stock, start="2010-01-01", end="2024-01-01")
                                                                                                                
        stock_data['Ticker'] = stock
        stock_data['RSI'] = calculate_rsi(stock_data['Close'], window=14)
        stock_data['RSI_Scaled'] = scaler.fit_transform(stock_data['RSI'].values.reshape(-1, 1))
        stock_data['Close_Scaled'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
        stock_data['High_Scaled'] = scaler.fit_transform(stock_data['High'].values.reshape(-1, 1))
        stock_data['Low_Scaled'] = scaler.fit_transform(stock_data['Low'].values.reshape(-1, 1))
        stock_data['Open_Scaled'] = scaler.fit_transform(stock_data['Open'].values.reshape(-1, 1))
        stock_data['Volume_Scaled'] = scaler.fit_transform(stock_data['Volume'].values.reshape(-1, 1))
        stock_data.reset_index(inplace=True)
        #stock_data = stock_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker', 'RSI', 'RSI_Scaled']]
        stock_data.columns = [col[0] for col in stock_data.columns]

        semiConductor_data = pd.concat([semiConductor_data, stock_data], ignore_index=True)

        #print([val[0] for val in semiConductor_data.columns.tolist()])
        #print(f"Total Data Length: {len(semiConductor_data)}")

    semiConductor_data = semiConductor_data.dropna()
    semiConductor_data.to_csv("semiconductorData.csv", index=False)
    print([ i for i in semiConductor_data['Ticker'].unique()])
    for stock in SEMICONDUCTORS:
        print(f"{stock} rows: {len(semiConductor_data.loc[semiConductor_data['Ticker'] == stock])}")

    #print(semiConductor_data.columns)
    semiConductor_data.drop(columns=['Date', 'RSI', 'Close', 'High', 'Low', 'Open', 'Volume'], inplace=True)
    #print(semiConductor_data.dtypes)
    #semiConductor_data.head(5)

def train_model():
    epochs = 100
    model = LSTMRegressionModel(input_size=6, output_size=10).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.L1Loss()

    losses = []  
    i = 0
    for ticker, loader in company_loaders.items(): 
        i += 1
        for epoch in range(epochs):
            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

                losses.append(loss.item())

                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Company: {ticker}, Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

        if i == 1:
            break
    torch.save(model.state_dict(), "LSTMRegression.pth")
    torch.save(model, "LSTMRegression.pth")

    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

def test():
    model.eval()

    test_loss = 0
    num_batches = 0

    with torch.no_grad():
        i = 0
        for ticker, loader in company_loaders.items():
            i += 1
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                y_pred = model(x_batch)
                
                loss = criterion(y_pred, y_batch)
                test_loss += loss.item()
                num_batches += 1
                
                if num_batches % 100 == 0:
                    print(f"Test Loss after {num_batches} batches: {test_loss / num_batches}")
            
            if i == 1:
                break

    avg_test_loss = test_loss / num_batches
    print(f"Average Test Loss: {avg_test_loss}")

    coordinates = np.linspace(start=0, stop=9, num=10, dtype=int).tolist()
    print(coordinates)

    ticker = "TSM" 
    j = 0
    for x_batch, y_batch in company_loaders[ticker]:
        j += 1
        
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        y_pred = model(x_batch)
        
        # Detach from the computation graph and move to CPU for plotting
        y_pred = y_pred.detach().cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        print(y_batch.shape)
        print(y_pred.shape)
        print(f"Predictions for {ticker}: {y_pred[0]}")
        print(f"True values for {ticker}: {y_batch[0]}")
        
        plt.plot(coordinates, y_batch[0], color="blue", label="True values")
        plt.plot(coordinates, y_pred[0], color="red", label="Predicted values")
        plt.legend()
        plt.show()

        if j == 10:
            break