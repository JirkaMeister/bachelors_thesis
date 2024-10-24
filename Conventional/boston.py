import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# Load and preprocess the data
housing = fetch_california_housing()
data = housing.data
target = housing.target

class ConventionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConventionalNeuralNetwork, self).__init__()

        self.fullyConnectedLayer1 = nn.Linear(8, 50)
        self.fullyConnectedLayer2 = nn.Linear(50, 50)
        self.fullyConnectedLayer3 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fullyConnectedLayer1(x))
        x = F.relu(self.fullyConnectedLayer2(x))
        x = self.fullyConnectedLayer3(x)
        return x
    
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(data, target, test_size = 0.2, random_state = 42)

# Standardize the data
scaler = StandardScaler()
xTrain = scaler.fit_transform(xTrain)
xTest = scaler.transform(xTest)

# Setting the device to either the GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Instantiating the model and moving it to either the GPU or CPU (device)
model = ConventionalNeuralNetwork().to(device)

# Convert the data to PyTorch tensors
xTrain = torch.FloatTensor(xTrain).to(device)
yTrain = torch.FloatTensor(yTrain).to(device)

# Define the loss function and the optimizer
lossFunction = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Train the model
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(xTrain)
    loss = lossFunction(output, yTrain.view(-1, 1))
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

def test():
    model.eval()

    with torch.no_grad():
        output = model(torch.FloatTensor(xTest).to(device))
        loss = lossFunction(output, torch.FloatTensor(yTest).view(-1, 1))
        print("Test Loss:", loss.item())

for epoch in range(100):
    train(epoch)
    test()

