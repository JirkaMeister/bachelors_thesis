import torch.nn.functional as F
import torch
import numpy as np
from net_model import NN
from dataset import train_data

# Custom funckce pro nastavení vah modelu z pole vah
def update_weights(model: NN, weights):
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        param.data.copy_(torch.from_numpy(weights[start:end].reshape(param.shape)))
        start = end
    

# Funkce pro výpočet fitness hodnoty (chyby modelu)
def fittness(model, weights, inputs, labels):
    with torch.no_grad():
        # Nastavení vah modelu
        update_weights(model, weights)
        # Výpočet predikce modelu a chyby
        predictions = model(inputs)
        error = F.mse_loss(predictions, labels)
    return error

# Funkce pro výběr nejlepších jedinců (podle fitness hodnot)
def selection(population, fittness_values):
    # Seřazení jedinců podle fitness hodnot (argsort vrací indexy, jak by měly být hodnoty seřazeny)
    indexes = np.argsort(fittness_values)
    # Vrácení poloviny nejlepších jedinců
    return population[indexes[:len(population) // 2]]

# Funkce pro křížení dvou jedinců
def crossover(parent1, parent2):
    # Náhodný bod křížení (např. 3)
    crossover_point = np.random.randint(1, len(parent1) - 1)
    # Vytvoření potomka
    # např. ######  ->   ###***
    #       ******       ***###
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

# Funkce pro mutaci jedince
def mutation(individual, mutation_rate):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            # Mutace hodnoty na náhodné číslo z intervalu <-1, 1>
            individual[i] = np.random.uniform(-1, 1)
    return individual

def genetic_algorithm(model: NN, inputs, labels, population_size=100, generations=1000, mutation_rate=0.01):
    # Generování náhodné populace
    # Každý jedinec je reprezentován polem vah modelu
    total_weights = sum(param.numel() for param in model.parameters())
    population = np.random.uniform(-1, 1, (population_size, total_weights))

    for generation in range(generations):
        # Výpočet fittness hodnoty pro každého jedince
        fittness_values = np.array([fittness(model, individual, inputs, labels) for individual in population])

        # Výběr nejlepších jedinců
        selected_population = selection(population, fittness_values)

        # Vytvoření nové populace
        new_population = []
        while len(new_population) < population_size:
            # Výběr dvou náhodných jedinců
            parent1, parent2 =  selected_population[np.random.randint(0, len(selected_population), size=2)]
            # Křížení a vznik potomků
            child1, child2 = crossover(parent1, parent2)
            # Mutace potomků
            child1, child2 = mutation(child1, mutation_rate), mutation(child2, mutation_rate)
            # Přidání potomků do nové populace
            new_population.append(child1)
            new_population.append(child2)

        population = np.array(new_population)

        # Nejlepší jedinec v generaci
        best_individual = np.min(fittness_values)
        print(f"Generation: {generation + 1}, Best fittness: {best_individual}")

        # Pokud je chyba menší než 0.01, ukončení algoritmu
        if best_individual < 0.01:
            break

    # Vrácení nejlepšího jedince
    return population[np.argmin(fittness_values)]


model = NN()

best_weights = genetic_algorithm(model, train_data["inputs"], train_data["labels"])

# Nastavení nejlepších vah modelu
update_weights(model, best_weights)

torch.save(model.state_dict(), "trainedModelGenetic.pth")