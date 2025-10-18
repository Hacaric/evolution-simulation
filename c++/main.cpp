#include <iostream>
#include <vector>
#include "neurons.h"
#include <cstdlib> // for rand(), srand()
#include <ctime>   // for time()
#include <random>  // For better random number generation

using namespace std;
typedef unsigned uint;

uint randint(uint range_start_included, uint range_end_excluded){
    // Using C++11 random for better distribution
    static std::mt19937 gen(time(0)); // Seed only once
    std::uniform_int_distribution<uint> distrib(range_start_included, range_end_excluded - 1);
    return distrib(gen);
}

uint randint(uint range_end_excluded){
    return randint(0, range_end_excluded);
}

nnfloat randfloat(nnfloat range_start, nnfloat range_end) {
    // Using C++11 random for uniform float distribution
    static std::mt19937 gen(time(0)); // Seed only once
    std::uniform_real_distribution<nnfloat> distrib(range_start, range_end);
    return distrib(gen);
}

vector<Network> mutate(Network parent, uint copy_amount, pair<uint, uint> mutations_per_network_range, pair<nnfloat, nnfloat> mutation_size_range){
    vector<Network> networks = vector<Network>();
    networks.reserve(copy_amount); // Pre-allocate memory to avoid reallocations

    for (uint i = 0; i < copy_amount; i++){
        // Create one copy
        Network child = parent.copy();

        // Mutate that single copy
        uint mutation_count = randint(mutations_per_network_range.first, mutations_per_network_range.second);
        for (uint mutation_i = 0; mutation_i < mutation_count; mutation_i++){
            if (child.connections.empty()) continue; // Avoid crash if there are no connections
            nnfloat mutation_size = randfloat(mutation_size_range.first, mutation_size_range.second);
            uint affected_connection_idx = randint(child.connections.size());
            child.connections[affected_connection_idx]->weight += mutation_size;
        }
        networks.push_back(move(child)); // Move the mutated child into the vector
    }
    return networks;
}

int main(){
    srand(time(NULL)); // Seed random number generator for rand()
    vector<Neuron*> neurons = vector<Neuron*>();
    for (uint i = 0; i < 5; i++){
        Neuron* current_neuron = new Neuron();
        neurons.push_back(current_neuron);
    }
    vector<Connection*> connections = vector<Connection*>();
    connections.push_back(new Connection(neurons[0], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[0], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[2], neurons[4], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[2], neurons[4], randfloat(-1, 1)));
    Network nn = Network(neurons, connections);
    vector<uint> input_neurons = vector<uint>();
    input_neurons.push_back(0);
    input_neurons.push_back(1);
    vector<nnfloat> input = vector<nnfloat>();
    input.push_back(randfloat(-1, 1));
    input.push_back(randfloat(-1, 1));
    vector<uint> output_neurons = vector<uint>();
    output_neurons.push_back(4);
    uint steps = 2;
    cout << "Output: " << nn.run(input_neurons, input, output_neurons, steps)[0] << endl;
    return 0;
}