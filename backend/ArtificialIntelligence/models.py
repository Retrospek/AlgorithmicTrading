import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

class LSTMRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMRegression, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, batch_first=True)
        self.dense1 = nn.Linear(256, 128)  
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, output_size)
            
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, (h_n, c_n) = self.lstm3(out)

        out = out[:, -1, :]

        out = self.dense1(out)  
        out = self.relu(out) 

        out = self.dense2(out)
        out = self.relu(out)

        out = self.dense3(out)
        
        return out
    


class LSTMdataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.data_values = self.data.values
        self.targetIDX = data.columns.get_loc('RSI')
        self.sequence_length = sequence_length
        
        self.valid_indices = []
        for i in range(len(self.data) - self.sequence_length + 1): 
            self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.valid_indices):
            raise IndexError("Index out of bounds")
        
        dayIDX = self.valid_indices[idx]
        
        history = self.data_values[dayIDX:dayIDX + self.sequence_length]
        
        if len(history) != self.sequence_length:
            raise ValueError(f"Inconsistent sequence length at index {idx}")

        history = torch.tensor(history, dtype=torch.float32)

        return history
