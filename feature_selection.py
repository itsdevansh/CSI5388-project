import pandas as pd
import numpy as np
import random
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def generate_X_y():
    num_rows = 100  # Number of rows
    num_cols = 80   # Number of columns
    # Generate a DataFrame with random 1s and 0s
    data = np.random.randint(2, size=(num_rows, num_cols))  # Random integers (0 or 1)
    columns = [f"Feature_{i+1}" for i in range(num_cols)]  # Column names: Feature_1, Feature_2, ...
    X = pd.DataFrame(data, columns=columns)  # Create DataFrame
    y = [random.randint(0, 1) for _ in range(num_rows)]
    return X, y

X, y = generate_X_y()

def _get_score_of_features_set(X, y):
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    scores_mean = np.mean(scores)
    return scores_mean

def random_search(X, y, all_features, max_iteration):
    best_features = []
    best_score = 0
    for i in range(max_iteration):
        random_features = random.sample(all_features, k=random.randint(7, 20)) # can change the range of this number
        score = _get_score_of_features_set(X[random_features], y)
        if score > best_score:
            best_score = score
            best_features = random_features
    return best_features

def _get_neighbor_status_for_sa(feature_list, all_features):
    features = feature_list
    # randomly add/remove one feature to/from the list
    choice = random.choice([0, 1])
    if len(feature_list) == 1:
        # cannot have an empty list
        choice = 1
    if choice == 1:
        flag = True
        while flag:
            new_feature = random.choice(all_features)
            if new_feature not in feature_list:
                features.append(new_feature)
                flag = False
    else:
        random_index = random.randrange(len(feature_list))
        features = feature_list.pop(random_index)
    return feature_list

def simulated_annealing(X, y, all_features, initial_temperature, min_temperature, cooling_rate, max_iteration):
    best_features = []
    best_score = 0
    temperature = initial_temperature

    current_features = random.sample(all_features, k=20) # randomly choose 20 features as the initial set
    current_score = _get_score_of_features_set(X[current_features], y)

    while temperature > min_temperature:
        print(f"[SA] Current temperature: {temperature}")
        for i in range(max_iteration):
            new_features = _get_neighbor_status_for_sa(current_features, all_features)
            new_score = _get_score_of_features_set(X[new_features], y)

            if new_score > current_score:
                current_features = new_features
                if new_score > best_score:
                    best_features = new_features
            else:
                probability = math.exp((current_score - best_score) / temperature) # metropolis criterion
                if random.random() < probability:
                    current_features = new_features

        temperature = temperature * cooling_rate
            
    return best_features



def _generate_initial_population(all_features, population_size):
    if population_size < 2:
        print("[Genetic search] Population size should be higher than 2.")
        return
    
    population = []
    for i in range(population_size):
        population.append(random.sample(all_features, k=random.randint(1, len(all_features)))) # can change the range of this number
    return population

def _crossover(parent1, parent2):
    max_length = max(len(parent1), len(parent2))
    crossover_index = random.randint(0, max_length-1)
    # sort the lists
    sorted_parents = sorted([parent1, parent2], key = len)
    if crossover_index < len(sorted_parents[0]):
        child1 = sorted_parents[0][:crossover_index] + sorted_parents[1][crossover_index:]
        child2 = sorted_parents[1][:crossover_index] + sorted_parents[0][crossover_index:]
    else:
        child1 = sorted_parents[0] + sorted_parents[1][crossover_index:]
        child2 = sorted_parents[1][:crossover_index]
    return child1, child2

def _sort_population(population):
    sorted_population_descending = sorted(population, key=lambda x: x['score'], reverse=True)
    parent1_index = crossover_index = random.randint(0, len(sorted_population_descending)//2)
    repeat = True
    while repeat:
        parent2_index = crossover_index = random.randint(0, len(sorted_population_descending)//2)
        if parent2_index != parent1_index:
            repeat = False
    return sorted_population_descending
    


def genetic_search(X, y, all_features, population_size, generations, mutation_rate):
    population_features = _generate_initial_population(all_features, population_size)
    population = []
    for features in population_features:
        population.append({'features': features, 'score': _get_score_of_features_set(X[features], y)})


    for generation in range(generations):
        print(f"[SA] Generation #{generation}")
        # sort the candidates in the population
        population = _sort_population(population)

        # crossover
        for i in range(0, population_size//2, 2):
            child1, child2 = _crossover(population[i]['features'], population[i+1]['features'])

            # mutation
            for child in [child1, child2]:
                if random.random() < mutation_rate:
                    child = _get_neighbor_status_for_sa(child, all_features) # randomly remove/add one feature

            population = population[:-2] # drop the two feature lists with the worst performance
            population.append({'features': child1, 'score': _get_score_of_features_set(X[child1], y)})
            population.append({'features': child2, 'score': _get_score_of_features_set(X[child2], y)})

        print(population[0]['features'])
        print(population[0]['score'])
        # print(population[-1]['features'])
        # print(population[-1]['score'])
        
    best_features = population[0]['features']
    return best_features

        



# all_features = X.columns.tolist()
# selected_features = random_search(X, y, all_features, 200)
# selected_features =  simulated_annealing(X, y, all_features, 500, 50, 0.9, 200)
# selected_features = genetic_search(X, y, all_features, 30, 200, 0.3)
# print(selected_features)
