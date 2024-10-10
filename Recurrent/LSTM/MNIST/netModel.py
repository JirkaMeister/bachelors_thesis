import torch.nn as NN
import torch.nn.functional as F
import torch.optim as optim
import torch

# Network's parameters
input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
root = "../../../data"

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelLSTM(NN.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ModelLSTM, self).__init__()
        # This lstm is provided by PyTorch
        self.lstm = NN.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        # Input is now in format (batch_size, sequence_length, input_size)

        # Fully connected layer
        self.fully_connected = NN.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        # Initial tensor for the cell state
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        # Forward pass through the rnn
        output, _ = self.lstm(x, (h0, c0))
        # Output: (batch_size, sequence_length, hidden_size)
        # We need output to be in the format (batch_size, hidden_size), so we remove the sequence_length dimension
        output = output[:, -1, :]
        output = self.fully_connected(output)
        return output