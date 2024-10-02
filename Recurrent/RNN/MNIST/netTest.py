import netModel as RNN
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import json

testData = datasets.MNIST(
    root = RNN.root,
    train = False,
    download = True,
    transform = ToTensor()
)

testLoader = DataLoader(testData, batch_size = 64, shuffle = True)

model = RNN.ModelRNN(RNN.input_size, RNN.hidden_size, RNN.num_layers, RNN.num_classes)
model.load_state_dict(torch.load("trainedNetwork.pth", weights_only = True))
model.to(RNN.device)

model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in testLoader:
        # Reshape the images to (sequence_length, input_size)
        images = images.reshape(-1, RNN.sequence_length, RNN.input_size).to(RNN.device)
        labels = labels.to(RNN.device)

        # Forward pass
        outputs = model(images)

        # Get the index of the output with the highest probability
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    with open("result.json", "w") as file:
        result = {"accuracy": accuracy}
        json.dump(result, file)