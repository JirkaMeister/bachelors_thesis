from imageClassNN import NeuralNetwork
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

# Types of images in the CIFAR-10 dataset
imageClasses = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testData = datasets.CIFAR10(
    root = "./data",
    train = False,
    download = True,
    transform = transformation
)

testLoader = DataLoader(testData, batch_size = 64, shuffle = True)

# Setting the device to either the GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Loading the neural network
net = NeuralNetwork().to(device)
net.load_state_dict(torch.load("trainedNetwork.pth", weights_only = True))

# Testing the neural network
correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testLoader:
        # Get the test data
        inputs, expectedOutput = data[0].to(device), data[1].to(device)
        # Forward pass
        outputs = net(inputs)
        # Get the index of the output with the highest probability
        _, predicted = torch.max(outputs.data, 1)
        # Count the number of input images
        total += expectedOutput.size(0)
        # Count the number of correct predictions
        correct += (predicted == expectedOutput).sum().item()

print(f"Accuracy: {100 * correct / total}%")

newTransformation = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test on custom image
def imageLoader(imagePath):
    image = Image.open(imagePath)
    image = newTransformation(image).float()
    image = image.unsqueeze(0)
    return image

imagePath = "data/cat.jpg"
image = imageLoader(imagePath).to(device)

net.eval()
with torch.no_grad():
    output = net(image)
    _, predicted = torch.max(output.data, 1)
    print(f"Predicted class: {imageClasses[predicted.item()]}")