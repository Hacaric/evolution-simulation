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

vector<Network> mutate(Network& parent, uint copy_amount, pair<uint, uint> mutations_per_network_range, pair<nnfloat, nnfloat> mutation_size_range, bool include_original){
    vector<Network> networks = vector<Network>();
    networks.reserve(copy_amount + uint(include_original)); // Pre-allocate memory to avoid reallocations
    if (include_original){
        networks.push_back(parent.copy());
    }

    for (uint i = 0; i < copy_amount; i++){
        // Create one copy
        Network child = parent.copy();

        // Mutate that single copy
        uint mutation_count = randint(mutations_per_network_range.first, mutations_per_network_range.second);
        for (uint mutation_i = 0; mutation_i < mutation_count; ++mutation_i){
            if (randint(0, 2) == 0){ // Mutate a connection weight
                if (child.connections.empty()) continue; // Avoid crash if there are no connections
                nnfloat mutation_size = randfloat(mutation_size_range.first, mutation_size_range.second);
                uint affected_connection_idx = randint(child.connections.size());
                child.connections[affected_connection_idx]->weight = sigmoid(child.connections[affected_connection_idx]->weight + mutation_size);
            } else {
                if (child.neurons.empty()) continue;
                nnfloat mutation_size = randfloat(mutation_size_range.first, mutation_size_range.second);
                uint affected_connection_idx = randint(child.neurons.size());
                child.neurons[affected_connection_idx]->bias = sigmoid(child.neurons[affected_connection_idx]->bias + mutation_size);
            }       
        }
        networks.push_back(move(child)); // Move the mutated child into the vector
    }
    return networks;
}

// nnfloat getNetworkCost(Network nn, uint steps, Dataset dataset)
Network* evolve(Network& nn, uint steps, Dataset& dataset, uint networks_per_iter, uint iterations){
    nnfloat min_cost = MAXNNFLOAT;
    nnfloat cost;
    Network parent_nn = nn.copy();

    for (uint iter = 0; iter < iterations; ++iter){
        min_cost = MAXNNFLOAT;
        int best_idx = -1;

        vector<Network> mutated_nn = mutate(parent_nn, networks_per_iter-1, {1,3}, {-1.0f, 1.0f}, true);
        for (uint i = 0; i < mutated_nn.size(); i++){
            // cout << "Mutation " << i << " cost: " << dataset.getNetworkCost(mutated_nn[i], steps) << endl;
            cost = dataset.getNetworkCost(mutated_nn[i], steps);
            if (cost < min_cost){
                min_cost = cost;
                best_idx = i;
            }
        }
        cout << "Lowest cost after " << iter << "th iteration: " << min_cost << endl; 
        parent_nn = mutated_nn[best_idx].copy();
    }
    return new Network(parent_nn); // Return a dynamically allocated copy
}

int main(){
    srand(time(NULL)); // Seed random number generator for rand()
    vector<Neuron*> neurons = vector<Neuron*>();
    for (uint i = 0; i < 5; i++){
        Neuron* current_neuron = new Neuron(randfloat(-1, 1));
        neurons.push_back(current_neuron);
    }
    vector<Connection*> connections = vector<Connection*>();
    connections.push_back(new Connection(neurons[0], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[0], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[2], neurons[4], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[2], neurons[4], randfloat(-1, 1)));
    vector<uint> input_neurons = vector<uint>();
    input_neurons.push_back(0);
    input_neurons.push_back(1);
    vector<nnfloat> input1 = vector<nnfloat>();
    input1.push_back(randfloat(-1, 1));
    input1.push_back(randfloat(-1, 1));
    vector<uint> output_neurons = vector<uint>();
    output_neurons.push_back(4);
    Network nn = Network(neurons, connections, input_neurons, output_neurons);
    uint steps = 2;

    vector<vector<nnfloat>> dataset_inputs = vector<vector<nnfloat>>();
    vector<vector<nnfloat>> dataset_labels = vector<vector<nnfloat>>();
    Dataset dataset = Dataset(2, 1); // Input size 2, output size 1 for XOR
    dataset.loadData_ByReference(dataset_inputs, dataset_labels);
    dataset.addEntry_ByReference({1.0f, 0.0f}, {0.0f});
    dataset.addEntry_ByReference({0.0f, 1.0f}, {0.0f});
    dataset.addEntry_ByReference({1.0f, 1.0f}, {1.0f});
    dataset.addEntry_ByReference({0.0f, 0.0f}, {1.0f});

    nnfloat cost = dataset.getNetworkCost(nn, steps);
    cout << "Initial Cost: " << cost << endl;

    uint iterations;
    uint copies_per_iter;
    cout << "Enter number of iterations: ";
    cin >> iterations;
    cout << "\nEnter number of copies_per_iter: ";
    cin >> copies_per_iter;
    Network* best_nn = evolve(nn, steps, dataset, copies_per_iter, iterations);

    cout << "Final (best) cost: " << dataset.getNetworkCost(*best_nn, steps) << endl;

    // Clean up the dynamically allocated memory
    delete best_nn;

    // cout << "Output: " << nn.run(input1, steps)[0] << endl;
    return 0;
}