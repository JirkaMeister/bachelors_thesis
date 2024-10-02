import netModel as LSTM
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt

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

test_data = data_list[LSTM.train_size:LSTM.train_size + LSTM.test_size]

# Create the datasets and dataloaders
test_dataset = TimeSeriesDataset(test_data, LSTM.sequence_length)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)

model = LSTM.ModelLSTM(LSTM.input_size, LSTM.hidden_size, LSTM.num_layers, 1)
model.load_state_dict(torch.load("trainedNetwork.pth", weights_only = True))
model.to(LSTM.device)

model.eval()
predictions = torch.tensor([], device=LSTM.device)
actuals = torch.tensor([], device=LSTM.device)

with torch.no_grad():
    for x, y in test_loader:
        x = x.reshape(-1, LSTM.sequence_length, LSTM.input_size).to(LSTM.device)
        y = y.to(LSTM.device)

        # Forward pass
        outputs = model(x)

        predictions = torch.cat((predictions, outputs), dim = 0)
        actuals = torch.cat((actuals, y.unsqueeze(-1)), dim = 0)

predictions = predictions.cpu().numpy()
actuals = actuals.cpu().numpy()

plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()