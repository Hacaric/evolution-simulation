from neural_network import *
from keras.datasets import mnist
import numpy

def difference(list1:list[int], list2:list[int]):
    if len(list1) != len(list2):
        raise Exception("Error in difference(): Lists must be of the same length")
        
    return sum([abs(list1[i] - list2[i])**2 for i in range(len(list1))])

def get_cost_of_nn(nn:NeuralNetwork, dataset:list[list[list[int], list[int]]]):
    results = []
    for input_, label in dataset:
        results.append(difference(nn.run(input_), label))
    return sum(results) / (len(dataset) * (nn.value_range[1] - nn.value_range[0]))

def random_float(range_:tuple[int, int]):
    random.seed(time.time())
    range_ = sorted(list(range_))
    return random.random()*(range_[1] - range_[0]) + range_[0]

def mutate(nn_parent:NeuralNetwork, mutation_rate:tuple[float, float], max_mutation_size:float):
    nn = nn_parent.copy()
    mutation_fraction = random_float(mutation_rate)
    mutation_size_weight = random_float((-max_mutation_size, max_mutation_size))
    mutation_size_bias = random_float((-max_mutation_size, max_mutation_size))
    total_weights_to_mutate = math.ceil(sum([connection.total_weights for connection in nn.connections]) * mutation_fraction)
    total_biases_to_mutate = math.ceil(sum([connection.total_biases for connection in nn.connections]) * mutation_fraction)
    for mutation_i in range(total_weights_to_mutate):
        connection = random.randint(0, len(nn.connections)-1)
        input_node_affected = random.randint(0, nn.connections[connection].input_size-1)
        output_node_affected = random.randint(0, nn.connections[connection].output_size-1)
        nn.connections[connection].weights[input_node_affected][output_node_affected] += mutation_size_weight
    for mutation_i in range(total_biases_to_mutate):
        output_node_affected = random.randint(0, nn.connections[connection].output_size-1)
        nn.connections[connection].biases[output_node_affected] += mutation_size_bias
    return nn
def mutate_2(nn_parent:NeuralNetwork, mutation_count:tuple[int, int], max_mutation_size:float):
    nn = nn_parent.copy()
    mutation_fraction = random_float(mutation_count)
    mutation_size_weight = random_float((-max_mutation_size, max_mutation_size))
    mutation_size_bias = random_float((-max_mutation_size, max_mutation_size))
    total_weights_to_mutate = random.randint(mutation_count[0], mutation_count[1])
    total_biases_to_mutate = random.randint(mutation_count[0], mutation_count[1])
    for mutation_i in range(total_weights_to_mutate):
        connection = random.randint(0, len(nn.connections)-1)
        input_node_affected = random.randint(0, nn.connections[connection].input_size-1)
        output_node_affected = random.randint(0, nn.connections[connection].output_size-1)
        nn.connections[connection].weights[input_node_affected][output_node_affected] += mutation_size_weight
    for mutation_i in range(total_biases_to_mutate):
        connection = random.randint(0, len(nn.connections)-1)
        output_node_affected = random.randint(0, nn.connections[connection].output_size-1)
        nn.connections[connection].biases[output_node_affected] += mutation_size_bias
    return nn