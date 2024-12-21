# Data Processing & Analysis
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    VotingRegressor, StackingRegressor, BaggingRegressor
)
import xgboost as xgb
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

# ===== Grabbing Stock list ===== #
from bs4 import BeautifulSoup
import requests
import re



def get_SPY_stocks():
    r = requests.get('https://stockanalysis.com/list/sp-500-stocks/')
    soup = BeautifulSoup(r.content, 'html.parser')
    # Main =>
    s = soup.find('main', class_='contain')
    # Find all tds with class "sym svelte-eurwtr"
    tds = s.find_all('td', class_='sym svelte-eurwtr')
    SPY = []
    for item in tds:
        # Find the text within the td
        ticker = item.find('a')
        ticker = ticker.text
        ticker = re.sub(r'[^a-zA-Z0-9]', '-', ticker)
        SPY.append(ticker)
    return SPY

SPY = get_SPY_stocks()
# ----- Query and Handle Data ----- #

def getTickerData(ticker, start="2023-01-01", end= date.today().strftime("%Y-%m-%d")):
    # Get the data from Yahoo Finance
    data = yf.download(tickers=ticker, start=start, end=end)

    # This will make day a column
    data.reset_index(inplace=True)

    data['Ticker'] = ticker
    
    return data

def process_df(df):
    """
    Calculate the percent change for a day
    Calculate the moving average for 20 and 30 day periods
    Calculate the momentum
    """
    # Calculate Daily % Change based on previous day's closing price
    df['Daily%Change'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)).round(4)

    df['5DaySMA'] = df['Daily%Change'].rolling(window=5).mean()
    df['10DaySMA'] = df['Daily%Change'].rolling(window=10).mean()

    return df



done = False
curr = []

totalData = pd.DataFrame(columns=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume", "Ticker", "Daily%Change", "5DaySMA", "10DaySMA"])

xgbRegressor = XGBRegressor()
randforRegressor = RandomForestRegressor()

while not done:
    ticker = input("Enter a ticker symbol (or 'q' to quit): ")
    if ticker.lower() == 'q':
        done = True

    data = process_df(getTickerData(ticker)).dropna()

    X = data.drop(columns=["Date", "Adj Close", "Close", "Volume", "Daily%Change", "Ticker"])
    y = data[["Close", "5DaySMA", "10DaySMA"]]
    X_train, y_train, X_test, y_test = train_test_split()




    
    










