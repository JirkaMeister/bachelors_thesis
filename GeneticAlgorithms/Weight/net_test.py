import torch.nn.functional as F
import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from net_model import NN
from dataset import test_data

model_genetic = NN()
model_genetic.load_state_dict(torch.load("trainedModelGenetic.pth", weights_only = True))
model_genetic.eval()

model_backpropagation = NN()
model_backpropagation.load_state_dict(torch.load("trainedModelBackpropagation.pth", weights_only = True))
model_backpropagation.eval()

# Výpočet predikce modelu
with torch.no_grad():
    predictions_genetic = model_genetic(test_data["inputs"])
    predictions_backpropagation = model_backpropagation(test_data["inputs"])
    # Výpočet přesnosti modelu
    mse_genetic = F.mse_loss(predictions_genetic, test_data["labels"].float().view(-1, 1))
    mse_backpropagation = F.mse_loss(predictions_backpropagation, test_data["labels"].float().view(-1, 1))
    print(f"Genetic algorithm model MSE: {mse_genetic.item()}")
    print(f"Backpropagation model MSE: {mse_backpropagation.item()}")