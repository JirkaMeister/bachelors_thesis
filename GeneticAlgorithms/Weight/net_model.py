
import torch.nn as neuralNetwork
import torch.nn.functional as F

class NN(neuralNetwork.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.fullyConnectedLayer1 = neuralNetwork.Linear(13, 50)
        self.fullyConnectedLayer2 = neuralNetwork.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fullyConnectedLayer1(x))
        x = self.fullyConnectedLayer2(x)
        return x