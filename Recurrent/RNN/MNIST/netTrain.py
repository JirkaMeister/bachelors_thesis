import netModel as RNN
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import torch


trainingData = datasets.MNIST(
    root = RNN.root,
    train = True,
    download = True,
    transform = ToTensor()
)

trainingLoader = DataLoader(trainingData, batch_size = 64, shuffle = True)


model = RNN.ModelRNN(RNN.input_size, RNN.hidden_size, RNN.num_layers, RNN.num_classes)
model.to(RNN.device)

optimizer = optim.Adam(model.parameters(), lr = RNN.learning_rate)

loss_function = nn.CrossEntropyLoss()


for epoch in range(RNN.num_epochs):
    print(f"Epoch: {epoch}")

    for i, (images, labels) in enumerate(trainingLoader):
        images = images.reshape(-1, RNN.sequence_length, RNN.input_size).to(RNN.device)
        labels = labels.to(RNN.device)

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "trainedNetwork.pth")