import netModel as LSTM
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
import torch


trainingData = datasets.MNIST(
    root = LSTM.root,
    train = True,
    download = True,
    transform = ToTensor()
)

trainingLoader = DataLoader(trainingData, batch_size = 64, shuffle = True)


model = LSTM.ModelLSTM(LSTM.input_size, LSTM.hidden_size, LSTM.num_layers, LSTM.num_classes)
model.to(LSTM.device)

optimizer = optim.Adam(model.parameters(), lr = LSTM.learning_rate)

loss_function = nn.CrossEntropyLoss()


for epoch in range(LSTM.num_epochs):
    print(f"Epoch: {epoch}")

    for i, (images, labels) in enumerate(trainingLoader):
        images = images.reshape(-1, LSTM.sequence_length, LSTM.input_size).to(LSTM.device)
        labels = labels.to(LSTM.device)

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "trainedNetwork.pth")