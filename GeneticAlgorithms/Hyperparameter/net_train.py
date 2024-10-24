import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from net_model_iris import dymamicNN
#from net_model_mnist import dymamicNN
from dataset_iris import test_data, train_data, INPUT_SIZE
#from dataset_mnist import test_loader, train_loader, INPUT_SIZE, OUTPUT_SIZE

EPOCHS = 3

def get_accuracy(model: dymamicNN):
    model.eval()
    with torch.no_grad():
        predicted = model(test_data["inputs"])
        _, predicted = torch.max(predicted, 1)
        accuracy = (predicted == test_data["labels"]).sum().item() / len(test_data["labels"])
    return accuracy

# Funkce pro výpočet fitness - params je list [počet vrstev, počet neuronů ve vrstvách, learning rate]
def fitness(individual):
    # Extrakce parametrů
    num_layers = int(individual[0])
    neurons_per_layer = int(individual[1])
    learning_rate = individual[2]

    # Definice modelu
    hidden_layers = [neurons_per_layer] * num_layers
    model = dymamicNN(INPUT_SIZE, hidden_layers, 3)

    # Definice optimizeru a loss funkce
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # Trénování modelu
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        output = model(train_data["inputs"])
        loss = loss_function(output, train_data["labels"])
        loss.backward()
        optimizer.step()

    # Výpočet accuracy na testovacích datech
    accuracy = get_accuracy(model)

    return 1 - accuracy

# Funkce pro selekci nejlepších jedinců
def selection(population, fitness_values):
    # Argsort vrátí indexy, jak by měly být hodnoty seřazeny
    indexes = np.argsort(fitness_values)
    # Vrácení poloviny nejlepších jedinců
    return population[indexes[:len(population) // 2]]

# Funkce pro křížení dvou jedinců
def crossover(parent1, parent2):
    # Náhodný bod křížení
    crossover_point = np.random.randint(0, len(parent1))
    # Vytvoření potomků
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

    return child1, child2

# Funkce pro mutaci jedince
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Mutace pro learning rate musí být v intervalu <0.001, 0.1>
            if i == 2:
                individual[i] += np.random.randn() * 0.01
                individual[i] = np.clip(individual[i], 0.001, 0.1)
            else:
                individual[i] += np.random.randn() * 0.1

    return individual

# Funkce pro genetický algoritmus - vrací parametry nejlepšího jedince
def genetic_algorithm(population_size=6, generations=100, mutation_rate=0.1):

    # Inicializace populace (num_hidden_layers, neurons_per_layer, learning_rate)
    population = np.random.rand(population_size, 3)
    population[:, 0] = np.random.randint(1, 5, size=population_size)  # num_hidden_layers (1-5)
    population[:, 1] = np.random.randint(5, 100, size=population_size)  # neurons_per_layer (5-100)
    population[:, 2] = np.random.uniform(0.001, 0.1, size=population_size)  # learning_rate (0.001 - 0.1)

    best_overall = None

    for generation in range(generations):
        # Výpočet fitness hodnoty pro každého jedince
        fitness_values = np.array([fitness(individual) for individual in population])

        # Výběr nejlepších jedinců
        selected_population = selection(population, fitness_values)

        # Vytvoření nové populace
        new_population = []
        while len(new_population) < population_size:
            # Výběr dvou náhodných jedinců
            parent1, parent2 = selected_population[np.random.randint(0, len(selected_population), size=2)]
            # Křížení a vznik potomků
            child1, child2 = crossover(parent1, parent2)
            # Mutace potomků
            child1, child2 = mutation(child1, mutation_rate), mutation(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        population = np.array(new_population)

        print(f"Generation: {generation + 1}, Best fitness: {np.min(fitness_values):.3f}")

    best_index = np.argmin(fitness_values)
    return population[best_index]

# Spuštění genetického algoritmu
best_params = genetic_algorithm()
print("Best params:")
print(f"\tNum hidden layers: {int(best_params[0])}")
print(f"\tNeurons per layer: {int(best_params[1])}")
print(f"\tLearning rate: {best_params[2]:.3f}")
print(f"\tAccuracy: {(1 - fitness(best_params)) * 100:.3f}%")