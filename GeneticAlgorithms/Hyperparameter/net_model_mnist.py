import torch.nn as nn

class dymamicNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(dymamicNN, self).__init__()
        
        layers = []

        # Vytvoření konvolučních vrstev
        layers.append(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())

        if len(hidden_layers) == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:

            # hidden_layers je list čísel, kolik neuronů má mít daná vrstva
            for hidden_layer in hidden_layers:
                layers.append(nn.Linear(input_size, hidden_layer))
                layers.append(nn.ReLU())
                input_size = hidden_layer

            # Výstupní vrstva
            layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Přidání všech vrstev do modelu
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
        
