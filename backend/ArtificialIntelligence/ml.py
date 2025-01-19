import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

class LSTMRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMRegression, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 256, batch_first=True)
        self.dense1 = nn.Linear(256, 256)  
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, output_size)
        

        self.dropout = nn.Dropout(0.3)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :] 
        #out = self.dropout(out)
        
        out = self.dense1(out)  
        out = self.relu(out) 

        out = self.dense2(out)
        out = self.relu(out)

        out = self.dense3(out)
        
        return out
    

class LSTMdataset(Dataset):
    def __init__(self, data, sequence_length, output_length):
        self.data = data
        #print(f"Data: {self.data}")
        #print(f"Length of Data {len(self.data)}")
        self.data_values = self.data.values
        self.targetIDX = data.columns.get_loc('RSI')
        self.sequence_length = sequence_length
        self.output_length = output_length
        self.sample_length = sequence_length + output_length
        
        
        self.valid_indices = []
        for i in range(len(self.data) - self.sample_length + 1): 
            self.valid_indices.append(i)
        #print(self.valid_indices)
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
