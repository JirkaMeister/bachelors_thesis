import netModel as GRU
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import torch


trainingData = datasets.MNIST(
    root = GRU.root,
    train = True,
    download = True,
    transform = ToTensor()
)

trainingLoader = DataLoader(trainingData, batch_size = 64, shuffle = True)


model = GRU.ModelGRU(GRU.input_size, GRU.hidden_size, GRU.num_layers, GRU.num_classes)
model.to(GRU.device)

optimizer = optim.Adam(model.parameters(), lr = GRU.learning_rate)

loss_function = nn.CrossEntropyLoss()


for epoch in range(GRU.num_epochs):
    print(f"Epoch: {epoch}")

    for i, (images, labels) in enumerate(trainingLoader):
        images = images.reshape(-1, GRU.sequence_length, GRU.input_size).to(GRU.device)
        labels = labels.to(GRU.device)

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "trainedNetwork.pth")