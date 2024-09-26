import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers because we are dealing with images
        # Stride = how much the kernel moves each time
        self.convolutionalLayer1 = nn.Conv2d(3, 12, 5) # (32 (image size) - 5 (kernel size)) / 1 (strides) + 1 = 28 => new image size is 28x28 with 12 channels (12, 28, 28)
        # Pooling layer to reduce the size of the image
        # The kernel of size 2x2 will slide over the image and take the maximum value in the 2x2 grid and output it to the new image into one pixel
        self.poolingLayer1 = nn.MaxPool2d(2, 2) # (28 (image size) - 2 (kernel size)) / 2 (strides) + 1 = 14 => new image size is 14x14 with 12 channels (12, 14, 14)
        self.convolutionalLayer2 = nn.Conv2d(12, 24, 5) # (14 (image size) - 5 (kernel size)) / 1 (strides) + 1 = 10 => new image size is 10x10 with 24 channels (24, 10, 10)
        # Can also just specify to use previous pooling layer instead of new one
        self.poolingLayer2 = nn.MaxPool2d(2, 2) # (10 (image size) - 2 (kernel size)) / 2 (strides) + 1 = 5 => new image size is 5x5 with 24 channels (24, 5, 5)

        # Fully connected layers
        # Flattening the tensor (24, 5, 5) => 24 * 5 * 5 (600 neurons)
        self.fullyConnectedLayer1 = nn.Linear(24 * 5 * 5, 120) # 600 neurons to 120 neurons
        self.fullyConnectedLayer2 = nn.Linear(120, 60) # 120 neurons to 60 neurons
        self.fullyConnectedLayer3 = nn.Linear(60, 10) # 60 neurons to 10 neurons (10 classes)

    def forward(self, x):
        # Passing the input tensor through the layers
        # Conv layer 1 -> Apply ReLu -> Pool layer -> Conv layer 2 -> Apply ReLu -> Pool layer
        x = F.relu(self.convolutionalLayer1(x))
        x = self.poolingLayer1(x)
        x = F.relu(self.convolutionalLayer2(x))
        x = self.poolingLayer2(x) # Can also just specify to use previous pooling layer instead of new one

        # Flattening the tensor
        x = x.view(-1, 24 * 5 * 5)
        # Passing the flattened tensor through the fully connected layers
        # Flatten -> Fully connected layer 1 -> Apply ReLu -> Fully connected layer 2 -> Apply ReLu -> Fully connected layer 3
        x = F.relu(self.fullyConnectedLayer1(x))
        x = F.relu(self.fullyConnectedLayer2(x))
        x = self.fullyConnectedLayer3(x)

        return x