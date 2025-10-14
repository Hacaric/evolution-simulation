from neural_network import *

def difference(list1:list[int], list2:list[int]):
    if len(list1) != len(list2):
        raise Exception("Error in difference(): Lists must be of the same length")
        
    return sum([abs(list1[i] - list2[i])**2 for i in range(len(list1))])

def evaluate_nn(nn:NeuralNetwork, dataset:list[list[list[int], list[int]]]):
    results = []
    for input_, answer in dataset:
        results.append(difference(nn.run(input_), answer))
    return sum(results)

def random_float(range_:tuple[int, int]):
    random.seed(time.time())
    range_ = sorted(list(range_))
    return random.random()*(range_[1] - range_[0]) + range_[0]

def mutate(nn_parent:NeuralNetwork, mutation_rate:tuple[float, float], max_mutation_size:float):
    nn = nn_parent.copy()
    mutation_prob = random_float(mutation_rate)
    mutation_size_weight = random_float((-max_mutation_size, max_mutation_size))
    mutation_size_bias = random_float((-max_mutation_size, max_mutation_size))
    for connection in nn.connections:
        for i in range(len(connection.weights)):
            for j in range(len(connection.weights[i])):
                if random.random() < mutation_prob:
                    connection.weights[i][j] = (connection.weights[i][j] + mutation_size_weight)
        for i in range(len(connection.biases)):
            if random.random() < mutation_prob:
                connection.biases[i] = (connection.biases[i] + mutation_size_bias)
    return nn


ITERATIONS = 500
NN_PER_ITERATION = 20
NN_MUTATION_RATE:tuple[float, float] = (0.05, 0.2)
NN_MAX_MUTATION_SIZE:float = 0.5

# --- 3x3 Pattern Recognition Dataset ---
# The network must learn to classify three different patterns on a 3x3 grid.
# Input: 9 neurons (3x3 grid flattened), Output: 3 neurons (one for each class)

# Pattern 1: Vertical Line -> [1, 0, 0]
v_line = [1,0,0, 1,0,0, 1,0,0]
v_line2 = [0,1,0, 0,1,0, 0,1,0]
v_line3 = [0,0,1, 0,0,1, 0,0,1]

# Pattern 2: Horizontal Line -> [0, 1, 0]
h_line = [1,1,1, 0,0,0, 0,0,0]
h_line2 = [0,0,0, 1,1,1, 0,0,0]
h_line3 = [0,0,0, 0,0,0, 1,1,1]

# Pattern 3: Diagonal Line -> [0, 0, 1]
d_line = [1,0,0, 0,1,0, 0,0,1]
d_line2 = [0,0,1, 0,1,0, 1,0,0]

dataset = [[v_line, [1,0,0]], [v_line2, [1,0,0]], [v_line3, [1,0,0]], [h_line, [0,1,0]], [h_line2, [0,1,0]], [h_line3, [0,0,1]], [d_line, [0,0,1]], [d_line2, [0,0,1]]]

NN_SIZE = [9, 10, 3] # 9 inputs, 1 hidden layer of 6, 3 outputs

def evolve(NN_SIZE, dataset, ITERATIONS, NN_PER_ITERATION, NN_MUTATION_RATE, NN_MAX_MUTATION_SIZE, debug = False) -> tuple[NeuralNetwork, int]:
    nn = NeuralNetwork(NN_SIZE)
    best_error_rate = 1
    for iter in range(ITERATIONS):
        nn_generation = [nn] # Elitism: carry over the best from the last generation
        for nn_idx in range(NN_PER_ITERATION):
            nn_generation.append(mutate(nn_generation[0], NN_MUTATION_RATE, (NN_MAX_MUTATION_SIZE)))
        error_rate = []
        for nn_idx in range(len(nn_generation)):
            error_rate.append(evaluate_nn(nn_generation[nn_idx], dataset))
        best_nn_idx = error_rate.index(min(error_rate))
        best_error_rate = error_rate[best_nn_idx]
        nn = nn_generation[best_nn_idx]
        if debug:
            print(f"Iteration {iter}: Best NN error = {error_rate[best_nn_idx]}")
        if error_rate[best_nn_idx] == 0:
            if debug:
                print(f"Error rate is 0: breaking learning")
            break
    return nn, error_rate[best_nn_idx]

best_error_rate_average = 0
print("Started testing...")
for test_i in range(100):
    nn, best_error_rate = evolve(NN_SIZE, dataset, ITERATIONS, NN_PER_ITERATION, NN_MUTATION_RATE, NN_MAX_MUTATION_SIZE, debug=False)
    best_error_rate_average += (best_error_rate * test_i + best_error_rate) / (test_i + 1)
    print(f"#{test_i}: Error rate after {ITERATIONS} iterations: {round(best_error_rate, 8):.8f}")

print(f"Best error rate average: {best_error_rate_average}")

print(f"Final NN output for [0,0]: {nn.run(dataset[0][0])} - Correct: {dataset[0][1]}")
print(f"Final NN output for [0,1]: {nn.run(dataset[1][0])} - Correct: {dataset[1][1]}")
print(f"Final NN output for [1,0]: {nn.run(dataset[2][0])} - Correct: {dataset[2][1]}")
# print(f"Final NN output for [1,1]: {nn.run(dataset[3][0])} - Correct: {dataset[3][1]}")
