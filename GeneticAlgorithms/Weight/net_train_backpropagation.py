import torch
import torch.nn as nn
import torch.optim as optim
from net_model import NN
from dataset import train_data

model = NN()

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(train_data["inputs"])
    loss = loss_function(output, train_data["labels"].float().view(-1, 1))
    loss.backward()
    optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

for epoch in range(100):
    train(epoch)

torch.save(model.state_dict(), "trainedModelBackpropagation.pth")