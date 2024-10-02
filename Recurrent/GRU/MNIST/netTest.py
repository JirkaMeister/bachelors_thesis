import netModel as GRU
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import json

testData = datasets.MNIST(
    root = GRU.root,
    train = False,
    download = True,
    transform = ToTensor()
)

testLoader = DataLoader(testData, batch_size = 64, shuffle = True)

model = GRU.ModelGRU(GRU.input_size, GRU.hidden_size, GRU.num_layers, GRU.num_classes)
model.load_state_dict(torch.load("trainedNetwork.pth", weights_only = True))
model.to(GRU.device)

model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in testLoader:
        # Reshape the images to (sequence_length, input_size)
        images = images.reshape(-1, GRU.sequence_length, GRU.input_size).to(GRU.device)
        labels = labels.to(GRU.device)

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