from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as neuralNetwork
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

trainingData = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

testData = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

print(trainingData.data.shape)

loaders = {
    "train": DataLoader(trainingData, batch_size = 64, shuffle = True),
    "test": DataLoader(testData, batch_size = 64, shuffle = True)
}

class ConventionalNeuralNetwork(neuralNetwork.Module):
    def __init__(self):
        super(ConventionalNeuralNetwork, self).__init__()

        self.convolutionalLayer1 = neuralNetwork.Conv2d(
            in_channels = 1,
            out_channels = 10,
            kernel_size = 5
        )
        self.convolutionalLayer2 = neuralNetwork.Conv2d(
            in_channels = 10,
            out_channels = 20,
            kernel_size = 5
        )
        # Randomly deactivate 25% of the neurons while training
        self.convolutionalLayer2Dropout = neuralNetwork.Dropout2d(0.25)

        self.fullyConnectedLayer1 = neuralNetwork.Linear(320, 50)
        self.fullyConnectedLayer2 = neuralNetwork.Linear(50, 10) # Needs to be 10 because there are 10 digits


    def forward(self, x):
        # Calling the relu activation function on the output of the convolutional layer
        x = F.relu(F.max_pool2d(self.convolutionalLayer1(x), 2))
        x = F.relu(F.max_pool2d(self.convolutionalLayer2Dropout(self.convolutionalLayer2(x)), 2))

        # Flattening the tensor before passing it to the fully connected layer
        x = x.view(-1, 320) #320 is the number of neurons in the tensor
        x = F.relu(self.fullyConnectedLayer1(x))
        x = F.dropout(x, training = self.training)
        x = self.fullyConnectedLayer2(x)

        return F.softmax(x, dim = 1)

# Setting the device to either the GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Instantiating the model and moving it to either the GPU or CPU (device)
model = ConventionalNeuralNetwork().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

lossFunction = neuralNetwork.CrossEntropyLoss()

def train(epoch):
    # Set the model to training mode
    model.train()
    # Iterate over the training data using the loader
    for batchIndex, (data, target) in enumerate(loaders["train"]):
        # Move the data and target to the device (GPU or CPU)
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Predict the output of the model and calculate the loss
        output = model(data)
        loss = lossFunction(output, target)

        # Backpropagate the loss - store the gradients in the model
        loss.backward()

        optimizer.step()

        if batchIndex % 10 == 0:
            print(f"Epoch: {epoch}, Batch Index: {batchIndex}, Loss: {loss.item()}")

def test():
    # Set the model to evaluation mode
    model.eval()

    testLoss = 0
    correct = 0

    # Disable gradient calculation
    with torch.no_grad():
        for data, target in loaders["test"]:
            # Move the data and target to the device (GPU or CPU)
            data, target = data.to(device), target.to(device)

            # Predict the output of the model given the data
            output = model(data)

            testLoss += lossFunction(output, target).item()

            # Get the index of the output with the highest probability
            prediction = output.argmax(dim = 1, keepdim = True)

            # Counter of correct predictions - checks all the predictions and adds the correct ones (ie. adding all the 1s)
            correct += prediction.eq(target.view_as(prediction)).sum().item() # Shape of target and prediction is different so we need to reshape target

    # Calculate the average loss
    testLoss /= len(loaders["test"].dataset)
    print(f"Test set: Average loss: {testLoss}, Accuracy: {correct}/{len(loaders['test'].dataset)} ({100. * correct / len(loaders['test'].dataset)}%)")

# Train the model for 10 epochs
for epoch in range(1, 11):
    train(epoch)
    test()

# Output 5 random predictions
for i in range(5):
    index = torch.randint(0, len(loaders["test"].dataset), (1,)).item()
    data = loaders["test"].dataset[index][0].to(device)
    output = model(data.unsqueeze(0))
    prediction = output.argmax(dim = 1).item()

    plt.imshow(data.cpu().squeeze().numpy(), cmap = "gray")
    plt.title(f"Prediction: {prediction}")
    plt.show()