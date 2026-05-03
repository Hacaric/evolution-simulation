from neural_network import *
from keras.datasets import mnist
import numpy
from evolve import *
import time


ITERATIONS = 5000
NN_PER_ITERATION = 5
NN_MUTATION_COUNT:tuple[float, float] = (1, 4)
NN_MAX_MUTATION_SIZE:float = 1
SAMPLE_SIZE_PER_GEN = 20
CHANGE_DATASET_SAMPLE_EVERY_X_ITERATIONS:int = 100
training_somples_count = 1000

print("Loading MNIST data...")
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print("Loaded MNIST data")
dataset:list[list[list[int], list[int]]] = []
NN_SIZE = [784, 16, 16, 10]
# random_pick = random.randint(0, 6000)
# for i in range(len(train_X[random_pick:random_pick+training_samples_count])):
#     image = train_X[i].flatten() / 255.0  # Normalize pixel values to [0, 1]
#     label = numpy.zeros(10)
#     label[train_y[i]] = 1
#     dataset.append([image.tolist(), label.tolist()])



def evolve(NN_SIZE, dataset, ITERATIONS, NN_PER_ITERATION, NN_MUTATION_RATE, NN_MAX_MUTATION_SIZE, SAMPLE_SIZE_PER_ITERATION, CHANGE_DATASET_SAMPLE_EVERY_X_ITERATIONS, debug = False, debug_round = "\b") -> tuple[NeuralNetwork, int]:
    nn = NeuralNetwork(NN_SIZE)
    best_cost = 1

    random_pick = random.randint(0, len(dataset) - SAMPLE_SIZE_PER_ITERATION - 1)
    dataset_sample = dataset[random_pick:random_pick + SAMPLE_SIZE_PER_ITERATION]
    for iter in range(ITERATIONS):
        iter_timestamp = time.time()
        if iter % CHANGE_DATASET_SAMPLE_EVERY_X_ITERATIONS == 0:
            random_pick = random.randint(0, len(dataset) - SAMPLE_SIZE_PER_ITERATION - 1)
            dataset_sample = dataset[random_pick:random_pick + SAMPLE_SIZE_PER_ITERATION]
        nn_generation = [nn] # Elitism: carry over the best from the last generation
        for nn_idx in range(NN_PER_ITERATION):
            nn_generation.append(mutate_2(nn_generation[0], NN_MUTATION_RATE, (NN_MAX_MUTATION_SIZE)))
        costs = []
        for nn_idx in range(len(nn_generation)):
            costs.append(get_cost_of_nn(nn_generation[nn_idx], dataset_sample))
        best_cost = min(costs)
        best_nn_idx = costs.index(best_cost)
        nn = nn_generation[best_nn_idx]
        if debug:
            print(f"#{debug_round} Iteration {iter}: Best NN cost = {costs[best_nn_idx]:.5f}   Took {round(time.time() - iter_timestamp,3):.3f}s")
        if costs[best_nn_idx] == 0:
            if debug:
                print(f"Cost is 0: breaking learning")
            break
    return nn, best_cost

best_cost_average = 0
nn:NeuralNetwork = None
test_count = 1
print("Started testing...")
training_start_timestamp = time.time()
try:
    for test_i in range(test_count):
        test_timestamp = time.time()
        # loading dataset
        dataset:list[list[list[int], list[int]]] = []
        random_pick = random.randint(0, 6000)
        for i in range(training_somples_count):
            image = train_X[i+random_pick].flatten() / 255.0  # Normalize pixel values to [0, 1]
            label = numpy.zeros(10)
            label[train_y[i+random_pick]] = 1
            dataset.append([image.tolist(), label.tolist()])
        #
        nn, best_cost = evolve(NN_SIZE, dataset, ITERATIONS, NN_PER_ITERATION, NN_MUTATION_COUNT, NN_MAX_MUTATION_SIZE, SAMPLE_SIZE_PER_GEN, CHANGE_DATASET_SAMPLE_EVERY_X_ITERATIONS, debug=True, debug_round = test_i)
        best_cost_average += best_cost
        print(f"#{test_i}: Costs after {ITERATIONS} iterations: {round(best_cost, 8):.8f}  ({round(time.time() - test_timestamp, 2)}s)")
except KeyboardInterrupt:
    print("\nInterupted...")
best_cost_average /= test_count

# for data in dataset:
#     input_, label = data
#     formatted_input = ""
#     for x in range(28):
#         for y in range(28):
#             formatted_input += "  " if input_[x*28 + y] < 0.5 else "EE"
#         formatted_input += "\n"
#     print(f"Final NN output for:\n{formatted_input}:\n{nn.run(input_)}, correct answer: \n{label}")

print(f"\nLast cost: {best_cost}")
print(f"Best cost average: {round(best_cost_average, 8):.8f}")
total_time = round(time.time() - training_start_timestamp, 3)
print(f"Training took {total_time} seconds - {total_time/ITERATIONS} per iteration.")

# import os
with open("neural_network_latest.json", "w") as f:
    data = nn.parse()
    f.write(data)
# Best cost average: 3.75666480 (rounds = 100, ITERATIONS = 100, NN_PER_ITERATION = 10, NN_MUTATION_RATE = (0.05, 0.5), NN_MAX_MUTATION_SIZE = 0.5, training data: 0 - 5)