import netModel as LSTM
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import torch
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + self.sequence_length]  # Predict the next value
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

csv = pd.read_csv("AMZN.csv")
data_list = csv["Close"].tolist()

train_data = data_list[:LSTM.train_size]

# Create the datasets and dataloaders
train_dataset = TimeSeriesDataset(train_data, LSTM.sequence_length)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = LSTM.ModelLSTM(LSTM.input_size, LSTM.hidden_size, LSTM.num_layers, 1)
model.to(LSTM.device)

optimizer = optim.Adam(model.parameters(), lr = LSTM.learning_rate)

loss_function = nn.MSELoss()


for epoch in range(LSTM.num_epochs):
    print(f"Epoch: {epoch}")

    for i, (x, y) in enumerate(train_loader):
        x = x.to(LSTM.device)
        y = y.to(LSTM.device).unsqueeze(-1)

        # Forward pass
        outputs = model(x)
        loss = loss_function(outputs, y)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "trainedNetwork.pth")