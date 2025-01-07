import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class lstmDataset(Dataset):
    def __init__(self, data, features, targets, input_window=5, output_window=10):
        self.data = torch.FloatTensor(data.values)
        self.input_window = input_window
        self.output_window = output_window
    
    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_window]
        y = self.data[idx + self.input_window : idx + self.input_window + self.output_window, 5]  # Assuming column 6 is the target
        return x, y

def create_loaders():
    semiConductor_data = pd.read_csv("backend\ArtificialIntelligence\semiconductorData.csv")
    features = ["Close", "High", "Low", "Open", "Volume"]
    targets = ["RSI_Scaled"]

    company_datasets = {}
    tickers = semiConductor_data['Ticker'].unique()

    # Create an lstmDataset for each company and store it in company_datasets
    for ticker in tickers:
        data = semiConductor_data[semiConductor_data['Ticker'] == ticker].drop(columns=["Ticker"])
        company_datasets[ticker] = lstmDataset(data, input_window=5, output_window=10, features=features, targets=targets)
    print(company_datasets)

    batch_size = 32 
    company_loaders = {}

    for ticker, dataset in company_datasets.items():
        company_loaders[ticker] = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(company_loaders)