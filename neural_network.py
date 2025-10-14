import random
import time
import math


class NNLayer:
    values = []
    size = -1
    def __init__(self, size, randomize=True, random_range=(-1,1)):
        self.values = []
        self.size = -1
        if randomize:
            weights = [self.random_float(random_range) for _ in range(size)]
        else:
            weights = [0 for _ in range(size)]

        self.values = weights
        self.size = len(weights)

    def load(self, data:list[int]):
        if len(data) != self.size:
            raise Exception(f"Error in NNLayer.load: data size ({len(data)}) does not match layer size ({self.size})")
            
        self.values = data

    def random_float(self, range_:tuple[int, int]):
        range_ = sorted(list(range_))
        return random.random()*(range_[1] - range_[0]) + range_[0]
        

class NNLayerConnections:
    input_size = -1
    output_size = -1
    weights = []
    biases = []
    value_range = (-1, -1)
    def __init__(self, input_layer_size, output_layer_size, randomize=True, random_range=(-1,1)):
        self.input_size = -1
        self.output_size = -1
        self.weights = []
        self.biases = []
        self.value_range = (-1, -1)
        if input_layer_size <= 0 or output_layer_size <= 0:
            raise Exception(f"Error in NNLayerConnections.__init__: input layer(={input_layer_size}) nor output layer(={output_layer_size}) can be <= 0!")
        self.input_size = input_layer_size
        self.output_size = output_layer_size
        self.value_range = tuple(sorted(list(random_range)))

        if randomize:
            self.weights = [[self.random_float(random_range) for _ in range(output_layer_size)] for _ in range(input_layer_size)]
            self.biases = [(self.random_float(random_range)) for _ in range(output_layer_size)]
        else:
            self.weights = [[0 for _ in range(output_layer_size)] for _ in range(input_layer_size)]
            self.biases = [0 for _ in range(output_layer_size)]

    def random_float(self, range_:tuple[int, int]):
        range_ = sorted(list(range_))
        return random.random()*(range_[1] - range_[0]) + range_[0]
        
    def connect_nodes(self, node1, node2, value):
        if node1 > self.input_size or node2 > self.output_size:
            raise Exception(f"Error in NNLayerConnections.connect_nodes: can't connect node {node1} and {node2} with value {value}: out of range - input size={self.input_size}, output size={self.input_size}")
        self.weights[node1][node2] = value
    
    def set_bias(self, node, value):
        if node > self.output_size:
            raise Exception(f"Error in NNLayerConnections.set_bias: can't set bias for node {node} with value {value}: out of range - output size={self.output_size}")
        self.biases[node] = value
    
    def load(self, weights:list[list[int]], biases:list[int]):
        if not (len(weights) == self.input_size and len(weights[0]) == self.output_size):
            raise Exception(f"Error in NNLayerConnections.load: can't load data with input size {len(weights)} and output size {len(weights[0])}")
        self.weights = weights

        if len(biases) != self.output_size:
            raise Exception(f"Error in NNLayerConnections.load: can't load biases with size {len(biases)}, expected {self.output_size}")
        self.biases = biases

class NeuralNetwork:
    temp_layers:NNLayer = []
    connections:NNLayerConnections = []
    def __init__(self, layer_sizes:list[int]):
        self.temp_layers = []
        self.connections = []
        if len(layer_sizes) < 2:
            raise Exception(f"Error in NeuralNetwork.__init__: A neural network must have at least 2 layers, but got {len(layer_sizes)}")
        elif 0 in layer_sizes:
            raise Exception(f"Error in NeuralNetwork.__init__: Layer sizes must be greater than 0, but got {layer_sizes}")
        
        self.temp_layers.append(NNLayer(layer_sizes[0]))
        for curr_layer_size in range(1, len(layer_sizes)):
            self.connections.append(NNLayerConnections(self.temp_layers[-1].size, layer_sizes[curr_layer_size]))
            self.temp_layers.append(NNLayer(layer_sizes[curr_layer_size]))
        
    def run(self, input_data:list[int]) -> list[int]:
        if len(input_data) != self.temp_layers[0].size:
            raise Exception(f"Error in NeuralNetwork.run: Input data size ({len(input_data)}) does not match the input layer size ({self.temp_layers[0].size})")
        self.temp_layers[0].load(input_data)

        for i in range(len(self.connections)):
            current_layer = self.temp_layers[i]
            next_layer = self.temp_layers[i+1]
            connections = self.connections[i]

            new_next_layer_values = [0] * next_layer.size

            for output_node_idx in range(next_layer.size):
                weighted_sum = 0
                for input_node_idx in range(current_layer.size):
                    weighted_sum += current_layer.values[input_node_idx] * connections.weights[input_node_idx][output_node_idx]
                new_next_layer_values[output_node_idx] = self.sigmoid(weighted_sum + connections.biases[output_node_idx])
            
            next_layer.load(new_next_layer_values)
        
        return [round(i, 3) for i in self.temp_layers[-1].values]
    
    @classmethod
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def copy(self):
        new_nn = NeuralNetwork([layer.size for layer in self.temp_layers])
        for i, connection_layer in enumerate(self.connections):
            deep_copied_weights = [inner_list.copy() for inner_list in connection_layer.weights]
            new_nn.connections[i].load(deep_copied_weights, connection_layer.biases.copy())
        return new_nn
        

def randomize_seed():
    random.seed(time.time())

# total = 0
# results = []
# for i in range(100):
#     nn = NeuralNetwork([2,1])
#     results.append(nn.run([0, 1])[0])
#     total += results[-1]
# print(f"Average: {total / 100}")
