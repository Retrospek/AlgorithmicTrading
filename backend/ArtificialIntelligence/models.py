import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

class LSTMRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super(LSTMRegressionModel, self).__init__()
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_size, 
                             hidden_size=64, 
                             batch_first=True, 
                             dropout=dropout)
        
        self.lstm2 = nn.LSTM(input_size=64, 
                             hidden_size=128, 
                             batch_first=True, 
                             dropout=dropout)

        self.lstm3 = nn.LSTM(input_size=128, 
                             hidden_size=64, 
                             batch_first=True, 
                             dropout=dropout)
        
        # Fully connected layers
        self.dense1 = nn.Linear(64, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, output_size)
        
        # Activation functions and dropout
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.15)
        
    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.relu(out)

        out, _ = self.lstm2(out)
        out = self.relu(out)

        out, (hn, _) = self.lstm3(out)  # Use the hidden state from the last LSTM
        out = self.relu(out)
        
        # Fully connected layers
        out = hn[-1]  # Final hidden state from the last LSTM layer
        out = self.dense1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.dense2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.dense3(out)  # Linear output for regression
        
        return out