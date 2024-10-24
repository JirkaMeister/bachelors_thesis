from torchvision import datasets

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

from imageClassNN import NeuralNetwork

import torchvision.transforms as transforms

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainingData = datasets.CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = transformation
)


# Data loader for efficient data loading
trainingLoader = DataLoader(trainingData, batch_size = 64, shuffle = True)

# Setting the device to either the GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Instantiating the model and moving it to either the GPU or CPU (device)
net = NeuralNetwork().to(device)

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

lossFunction = nn.CrossEntropyLoss()

# Training the neural network
for epoch in range(30):
    print(f"Epoch: {epoch}")

    runningLoss = 0.0
    print(f"Length of training data loader: {len(trainingLoader)}")
    for i, data in enumerate(trainingLoader):
        inputs, expectedOutput = data[0].to(device), data[1].to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = net(inputs)
        # Calculate the loss
        loss = lossFunction(outputs, expectedOutput)
        # Backpropagation
        loss.backward()
        # Update the weights
        optimizer.step()

        runningLoss += loss.item()

    print(f"Loss: {runningLoss / len(trainingLoader)}")

torch.save(net.state_dict(), "trainedNetwork.pth")