import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wine = load_wine()
inputs = wine['data']
labels = wine['target']

# Normalizace dat
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Rozdělení dat na trénovací a testovací
xTrain, xTest, yTrain, yTest = train_test_split(inputs, labels, test_size = 0.2, random_state = 42)

# Převod dat na PyTorch tensory
xTrain = torch.FloatTensor(xTrain)
xTest = torch.FloatTensor(xTest)
yTrain = torch.LongTensor(yTrain)
yTest = torch.LongTensor(yTest)

train_data = {
    "inputs": xTrain,
    "labels": yTrain
}

test_data = {
    "inputs": xTest,
    "labels": yTest
}