from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import torch

# Načtení datasetu
iris = load_iris()
inputs = iris['data']
targets = iris['target']

# Normalizace dat
scaler = StandardScaler()
inputs = scaler.fit_transform(inputs)

# Rozdělení na trénovací a testovací sadu
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

inputs_train = torch.tensor(inputs_train, dtype=torch.float32)
targets_train = torch.tensor(targets_train, dtype=torch.long)
inputs_test = torch.tensor(inputs_test, dtype=torch.float32)
targets_test = torch.tensor(targets_test, dtype=torch.long)

train_data = {
    "inputs": inputs_train,
    "labels": targets_train
}

test_data = {
    "inputs": inputs_test,
    "labels": targets_test
}

INPUT_SIZE = train_data["inputs"].shape[1]