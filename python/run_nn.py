from neural_network import *
from keras.datasets import mnist
import numpy
import evolve
import time
file = "neural_network_latest.json"
random.seed(time.time())

with open("neural_network_latest.json", "r") as f:
    data = f.read()
nn = NeuralNetwork.load_parsed_data(data)

testing_samples_count = 5

print("Loading MNIST data...")
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print("Loaded MNIST data")
dataset:list[list[list[int], list[int]]] = []
random_pick = random.randint(0, 6000)
for i in range(testing_samples_count):
    image = train_X[i + random_pick].flatten() / 255.0  # Normalize pixel values to [0, 1]
    label = numpy.zeros(10)
    label[train_y[i + random_pick]] = 1
    dataset.append([image.tolist(), label.tolist()])

for data in dataset:
    input_, label = data
    formatted_input = ""
    for x in range(28):
        for y in range(28):
            formatted_input += "  " if input_[x*28 + y] < 0.5 else "EE"
        formatted_input += "\n"
    print(f"NN output for:\n{formatted_input}:\n{", ".join([str(i) + "0"*(5 - len(str(i))) for i in nn.run(input_)])}, correct answer: \n{", ".join([str(i) + "0"*(5 - len(str(i))) for i in label])}")
print(f"\nCost: {evolve.get_cost_of_nn(nn, dataset)}")
