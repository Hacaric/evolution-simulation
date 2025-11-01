#include <iostream>
#include <vector>
#include "neurons.h"
#include <cstdlib> // for rand(), srand()
#include <ctime>   // for time()
#include <random>  // For better random number generation
#include <algorithm> // For std::sort

using namespace std;
typedef unsigned uint;


bool comp_function_of__sort_by_cost(pair<nnfloat, uint> a, pair<nnfloat, uint> b){
    return a.first < b.first;
}
vector<Network> sort_by_cost(vector<Network>& networks, Dataset dataset, uint steps, uint keep_max=-1) {
    vector<pair<nnfloat, uint>> cost_to_network = vector<pair<nnfloat, uint>>();
    for (uint nn_i = 0; nn_i < networks.size(); nn_i++){
        cost_to_network.push_back(make_pair(dataset.getNetworkCost(networks[nn_i], steps), nn_i));
    }
    std::sort(cost_to_network.begin(), cost_to_network.end(), comp_function_of__sort_by_cost);
    vector<Network> networks2 = vector<Network>();
    for (uint nn_i = 0; nn_i < networks.size() && nn_i < keep_max; nn_i++){
        networks2.push_back(networks[cost_to_network[nn_i].second].copy());
    }
    return networks2;
}

nnfloat max(nnfloat a, nnfloat b){
    return a > b ? a : b;
}
long max(long a, long b){
    return a > b ? a : b;
}

uint randint(uint range_start_included, uint range_end_excluded){
    if (range_start_included > range_end_excluded){
        throw out_of_range("range_start_included > range_end_excluded");
    }
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

vector<Network> mutate(Network& parent, uint copy_amount, bool include_original){
    vector<Network> networks = vector<Network>();
    networks.reserve(copy_amount + uint(include_original)); // Pre-allocate memory to avoid reallocations
    if (include_original){
        networks.push_back(parent.copy());
    }

    for (uint i = 0; i < copy_amount; i++){
        // Create one copy
        Network child = parent.copy();

        // Mutate that single copy
        cout << "Mutating, min: " << parent.mutations_per_copy_range.first << endl;
        if (parent.mutations_per_copy_range.first > 100000){
            throw out_of_range("AAAAAA :D");
        }
        uint mutation_count = randint(parent.mutations_per_copy_range.first, parent.mutations_per_copy_range.second);
        for (uint mutation_i = 0; mutation_i < mutation_count; ++mutation_i){
            nnfloat mutation_type = randfloat(0, parent.mutation_type_probabilities_sum);
            // cout << "Mutation type: " << mutation_type << endl;


            mutation_type -= parent.mutation_type_probabilities[0];
            if (mutation_type <= 0){ 
                // cout << "Mutation type chosen: Weight mutation" << endl;
                if (child.connections.empty()) continue; // Mutation of a weight
                nnfloat mutation_size = randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second);
                uint affected_connection_idx = randint(child.connections.size());
                child.connections[affected_connection_idx]->weight = child.connections[affected_connection_idx]->weight + mutation_size;
                continue;
            } 

            mutation_type -= parent.mutation_type_probabilities[1];
            if (mutation_type <= 0) { // Mutation of a bias
                // cout << "Mutation type chosen: Bias mutation" << endl;
                if (child.neurons.empty()) continue;
                nnfloat mutation_size = randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second);
                uint affected_connection_idx = randint(child.neurons.size());
                child.neurons[affected_connection_idx]->bias = child.neurons[affected_connection_idx]->bias + mutation_size;
                continue;
            }

            mutation_type -= parent.mutation_type_probabilities[2];
            if (mutation_type <= 0) { // Mutation of 'mutations_per_copy_range' span
                uint mutation_range_lower_bound = child.mutations_per_copy_range.first;
                long mutation_range_span = child.mutations_per_copy_range.second - child.mutations_per_copy_range.first;
                long curr_mutation_size = round(randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second));
                
                mutation_range_span += curr_mutation_size;
                if (mutation_range_span < 0) mutation_range_span = 0;

                child.mutations_per_copy_range.second = mutation_range_lower_bound + (uint)mutation_range_span;
                continue;
            }

            mutation_type -= parent.mutation_type_probabilities[3];
            if (mutation_type <= 0){ // Mutation of 'mutations_per_copy_range' lower bound
                long mutation_range_lower_bound = child.mutations_per_copy_range.first;
                long mutation_range_span = child.mutations_per_copy_range.second - child.mutations_per_copy_range.first;
                long curr_mutation_size = round(randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second));
                
                mutation_range_lower_bound += curr_mutation_size;
                if (mutation_range_lower_bound < 0) mutation_range_lower_bound = 0;

                child.mutations_per_copy_range.first = (uint)mutation_range_lower_bound;
                child.mutations_per_copy_range.second = (uint)mutation_range_lower_bound + (uint)mutation_range_span;
                continue;
            }

            mutation_type -= parent.mutation_type_probabilities[4];
            if (mutation_type <= 0) { // Mutation of 'mutations_size_range' span
                // cout << "Mutation type chosen: mutations_size_range span mutation" << endl;
                nnfloat mutation_size_lower_bound = child.mutations_size_range.first;
                nnfloat mutation_size_span = child.mutations_size_range.second - child.mutations_size_range.first;
                nnfloat curr_mutation_size = randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second) / 4; // Slow down mutation of mutation size
                mutation_size_span += max(curr_mutation_size, nnfloat(0));
                child.mutations_size_range.second = mutation_size_lower_bound + mutation_size_span;
                continue;
            }

            mutation_type -= parent.mutation_type_probabilities[5];
            if (mutation_type <= 0){ // Mutation of 'mutations_size_range' lower bound
                // cout << "Mutation type chosen: mutations_size_range lower bound mutation" << endl;
                nnfloat mutation_size_lower_bound = child.mutations_size_range.first;
                nnfloat mutation_size_span = child.mutations_size_range.second - child.mutations_size_range.first;
                nnfloat curr_mutation_size = randfloat(parent.mutations_size_range.first, parent.mutations_size_range.second) / 4; // Slow down mutation of mutation size
                mutation_size_lower_bound += max(curr_mutation_size, nnfloat(0));
                child.mutations_size_range.first = mutation_size_lower_bound;
                child.mutations_size_range.second = mutation_size_lower_bound + mutation_size_span;
                continue;
            }
        }
        networks.push_back(move(child)); // Move the mutated child into the vector
    }
    return networks;
}

// nnfloat getNetworkCost(Network nn, uint steps, Dataset dataset)
Network* evolve(Network& nn, uint steps, Dataset& dataset, uint networks_per_iter, uint iterations, uint amount_of_networks_carried_to_next_round){
    // nnfloat min_cost = MAXNNFLOAT;
    // nnfloat cost;
    vector<Network> parents = vector<Network>();
    parents.push_back(nn.copy());
    vector<Network> mutated_nn = vector<Network>();
    // Network parent_nn = nn.copy();

    for (uint iter = 0; iter < iterations; ++iter){
        // min_cost = MAXNNFLOAT;
        // int best_idx = -1;
        mutated_nn = vector<Network>();

        uint kid_count_rounded_up = ceil(networks_per_iter / amount_of_networks_carried_to_next_round);
        uint kid_count_rounded_down = floor(networks_per_iter / amount_of_networks_carried_to_next_round);
        
        // Think of this like a grid with sides 'amount_of_networks_carried_to_next_round' and 'kid_count_rounded_up' filled with networks. It's completely filled except for the last row. How do you get amount of filled sqares in the last row? X * Y - Content = free sqares => X - free sqares = amount of filled sqares in the last row
        // so its X - (X*Y - Content) = amount of filled sqares in the last row (= X*(1-Y)+Content)
        // X = amount_of_networks_carried_to_next_round
        // Y = 'kid_count_rounded_up' = ceil(networks_per_iter / amount_of_networks_carried_to_next_round) 
        // Content = networks_per_iter
        uint number_of_networks_that_should_have__kid_count_rounded_up = amount_of_networks_carried_to_next_round - (kid_count_rounded_up * amount_of_networks_carried_to_next_round - networks_per_iter);
        
        for (uint parent_nn_i = 0; parent_nn_i < parents.size(); parent_nn_i++){
            vector<Network> curr_mutated;
            if (parent_nn_i < number_of_networks_that_should_have__kid_count_rounded_up){
                curr_mutated = mutate(parents[parent_nn_i], kid_count_rounded_up, true);
            } else {
                curr_mutated = mutate(parents[parent_nn_i], kid_count_rounded_down, true);
            }
            mutated_nn.insert(mutated_nn.end(), curr_mutated.begin(), curr_mutated.end()); // https://www.geeksforgeeks.org/cpp/concatenate-two-vectors-in-cpp/
        }    
        parents = sort_by_cost(mutated_nn, dataset, steps, amount_of_networks_carried_to_next_round);
        // for (uint i = 0; i < mutated_nn.size(); i++){
        //     // cout << "Mutation " << i << " cost: " << dataset.getNetworkCost(mutated_nn[i], steps) << endl;
        //     cost = dataset.getNetworkCost(mutated_nn[i], steps);
        //     if (cost < min_cost){
        //         min_cost = cost;
        //         best_idx = i;
        //     }
        // }
        cout.precision(8); cout << fixed << "Lowest cost after " << iter << "th iteration: " << dataset.getNetworkCost(parents[0], steps) << endl; 
        // parent_nn = mutated_nn[best_idx].copy();
    }
    return new Network(parents[0].copy()); // Return a dynamically allocated deep copy
}

int main(){

    vector<nnfloat> mutation_type_probabilities = vector<nnfloat>(
        {
            70.0f, // Mutation of a weight
            20.0f, // Mutation of a bias
            2.5f,  // Mutation of 'mutation_range' span (e.g. original={1, 8} mutated={1, 5})
            2.5f,  // Mutation of 'mutation_range' start (e.g. original={1, 8} mutated={0, 7})
            2.5f,  // Mutation of 'mutation_size' span (e.g. original={1, 8} mutated={1, 5})
            2.5f,  // Mutation of 'mutation_size' start (e.g. original={1, 8} mutated={0, 7})
        }
    );

    uint iterations;
    uint copies_per_iter;
    uint amount_of_networks_carried_to_next_round;
    cout << "Enter number of iterations: ";
    cin >> iterations;
    cout << "Enter number of copies_per_iter: ";
    cin >> copies_per_iter;
    cout << "Enter amount of networks carried to next round: ";
    cin >> amount_of_networks_carried_to_next_round;

    srand(time(NULL)); // Seed random number generator for rand()
    vector<Neuron*> neurons = vector<Neuron*>();
    for (uint i = 0; i < 6; i++){
        Neuron* current_neuron = new Neuron(randfloat(-1, 1));
        neurons.push_back(current_neuron);
    }
    vector<Connection*> connections = vector<Connection*>();
    connections.push_back(new Connection(neurons[0], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[2], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[0], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[3], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[0], neurons[4], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[1], neurons[4], randfloat(-1, 1)));

    connections.push_back(new Connection(neurons[2], neurons[5], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[3], neurons[5], randfloat(-1, 1)));
    connections.push_back(new Connection(neurons[4], neurons[5], randfloat(-1, 1)));
    vector<uint> input_neurons = vector<uint>();
    input_neurons.push_back(0);
    input_neurons.push_back(1);
    vector<nnfloat> input1 = vector<nnfloat>();
    input1.push_back(randfloat(-1, 1));
    input1.push_back(randfloat(-1, 1));
    vector<uint> output_neurons = vector<uint>();
    output_neurons.push_back(5);
    pair<uint, uint> mutations_per_copy_range = {4,10};
    pair<nnfloat, nnfloat> mutation_size_range = {-0.5f, 0.5f};
    Network nn = Network(neurons, connections, input_neurons, output_neurons, mutations_per_copy_range, mutation_size_range, mutation_type_probabilities);
    uint steps = 2;

    vector<vector<nnfloat>> dataset_inputs = vector<vector<nnfloat>>();
    vector<vector<nnfloat>> dataset_labels = vector<vector<nnfloat>>();
    Dataset dataset = Dataset(2, 1); // Input size 2, output size 1 for XOR
    dataset.loadData_ByReference(dataset_inputs, dataset_labels); // Using ByReference to avoid copying
    // XNOR-like patterns
    dataset.addEntry_ByCopying({1.0f, 1.0f}, {1.0f});
    dataset.addEntry_ByCopying({0.0f, 0.0f}, {1.0f});
    dataset.addEntry_ByCopying({-1.0f, -1.0f}, {1.0f});
    dataset.addEntry_ByCopying({1.0f, 0.9f}, {1.0f});

    // XOR-like patterns
    dataset.addEntry_ByCopying({1.0f, 0.0f}, {0.0f});
    dataset.addEntry_ByCopying({0.0f, 1.0f}, {0.0f});
    dataset.addEntry_ByCopying({-1.0f, 1.0f}, {0.0f});
    dataset.addEntry_ByCopying({1.0f, -1.0f}, {0.0f});
    dataset.addEntry_ByCopying({0.1f, 0.9f}, {0.0f});

    nnfloat cost = dataset.getNetworkCost(nn, steps);
    cout << "Initial Cost: " << cost << endl;

    Network* best_nn = evolve(nn, steps, dataset, copies_per_iter, iterations, amount_of_networks_carried_to_next_round);

    cout << "Final (best) cost: " << dataset.getNetworkCost(*best_nn, steps) << endl;
    for (uint input_i = 0; input_i < dataset.inputs.size(); input_i++){
        for (uint i = 0; i < dataset.inputs[input_i].size(); i++){
            cout << fixed << dataset.inputs[input_i][i] << ", ";
        }
        cout << "\b\b: network output: " << fixed << best_nn->run(dataset.inputs[input_i], steps)[0] << " correct answer: " << dataset.labels[input_i][0] << endl;
    }
    // Clean up the dynamically allocated memory
    delete best_nn;

    // cout << "Output: " << nn.run(input1, steps)[0] << endl;
    return 0;
}