import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class StockDataPreprocessor:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def calculate_returns(self, prices):
        """Calculate log returns"""
        return np.log(prices / prices.shift(1))
    
    def calculate_volatility(self, returns, window=20):
        """Calculate rolling volatility"""
        return returns.rolling(window=window).std()
    
    def normalize_volume(self, volume, window=20):
        """Normalize volume using rolling statistics"""
        rolling_mean = volume.rolling(window=window).mean()
        rolling_std = volume.rolling(window=window).std()
        return (volume - rolling_mean) / rolling_std
    
    def process_data(self):
        all_data = []
        
        for ticker in self.tickers:
            # Download data
            stock_data = yf.download(ticker, start=self.start_date, end=self.end_date)
            
            # Calculate price-based features
            stock_data['Returns'] = self.calculate_returns(stock_data['Close'])
            stock_data['Volatility'] = self.calculate_volatility(stock_data['Returns'])
            
            # Normalize volume relative to its recent history
            stock_data['NormalizedVolume'] = self.normalize_volume(stock_data['Volume'])
            
            # Calculate price ratios (these are already normalized)
            stock_data['HL_Ratio'] = stock_data['High'] / stock_data['Low'] - 1
            stock_data['CO_Ratio'] = stock_data['Close'] / stock_data['Open'] - 1
            
            # Calculate RSI (already normalized 0-100)
            stock_data['RSI'] = calculate_rsi(stock_data['Close'])
            
            # Add ticker information
            stock_data['Ticker'] = ticker
            stock_data.reset_index(inplace=True)
            
            all_data.append(stock_data)
            
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Drop rows with NaN values (from rolling calculations)
        combined_data.dropna(inplace=True)
        
        return combined_data

class LSTMDataset(Dataset):
    def __init__(self, data, sequence_length, forecast_horizon):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Use normalized features
        self.feature_columns = [
            'Returns', 'Volatility', 'NormalizedVolume',
            'HL_Ratio', 'CO_Ratio', 'RSI'
        ]
        
        # Group by ticker to ensure sequences don't cross between stocks
        self.grouped_data = dict(tuple(data.groupby('Ticker')))
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        for ticker, group in self.grouped_data.items():
            features = group[self.feature_columns].values
            for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
                x = features[i:i+self.sequence_length]
                y = features[i+self.sequence_length:i+self.sequence_length+self.forecast_horizon, -1]  # RSI values
                samples.append((x, y))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Self-attention on the sequence
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)
        
        # Residual connection and layer norm
        combined = self.layer_norm(lstm_out + attn_out)
        
        # Take the last sequence element
        out = combined[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        
        return out

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

# Example usage
if __name__ == "__main__":
    # Configuration
    TICKERS = ["NVDA", "TSM", "AVGO", "AMD", "ASML", "MRVL", "ON"]
    SEQUENCE_LENGTH = 30
    FORECAST_HORIZON = 5
    BATCH_SIZE = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Prepare data
    preprocessor = StockDataPreprocessor(TICKERS, '2010-01-01', '2024-01-01')
    data = preprocessor.process_data()
    
    # Create dataset
    dataset = LSTMDataset(data, SEQUENCE_LENGTH, FORECAST_HORIZON)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = LSTMForecaster(
        input_size=6,  # number of features after preprocessing
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=FORECAST_HORIZON
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, NUM_EPOCHS, device
    )