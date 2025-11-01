#include <iostream>
#include <vector>
#include <unordered_map>
#include <math.h>
#include "neurons.h"

using namespace std;

nnfloat sigmoid(nnfloat x) {
    // return x;
    return (2.0f / (1.0f + exp(-x))) - 1.0f;
}


// struct Neuron{
//     nnfloat temp_value; nnfloat temp_value2;
//     vector<Neuron*> connections;
//     Neuron(){
//         connections = vector<Neuron*>();
//     };
// };

// struct Connection{
//     Neuron* source;
//     Neuron* target;
//     nnfloat weight;
//     Connection(Neuron*, Neuron*, nnfloat);
// };

Connection::Connection(Neuron* neuron1_, Neuron* neuron2_, nnfloat weight_){
    source = neuron1_;
    target = neuron2_;
    weight = weight_;
};

// class Network{
// private:
//     vector<Neuron*> neurons;
//     vector<Connection*> connections;
//     vector<vector<Connection*>> map;
//     unordered_map<Neuron*, uint> map_neuron_to_index; 
// public:
//     Network(vector<Neuron*>, vector<Connection*>);
//     vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat) = sigmoid);
// };

Network::Network(vector<Neuron*> neurons_, vector<Connection*> connections_, vector<uint> default_input_neurons_, vector<uint> default_output_neurons_, pair<uint, uint> mutations_per_copy_range_, pair<nnfloat, nnfloat> mutation_size_range_, vector<nnfloat> mutation_type_probabilities_){
    neurons = neurons_;
    connections = connections_;
    default_input_neurons = vector<uint>(default_input_neurons_);
    default_output_neurons = vector<uint>(default_output_neurons_);
    mutations_per_copy_range = mutations_per_copy_range_;
    mutations_size_range = mutation_size_range_;


    if (mutation_type_probabilities_.size() != MUTATION_TYPES_COUNT){
        throw invalid_argument("Error: mutation_type_probabilities_ must have " + to_string(MUTATION_TYPES_COUNT) + " elements.");
    }
    mutation_type_probabilities = vector<nnfloat>(mutation_type_probabilities_);

    if (mutation_type_probabilities.size() != MUTATION_TYPES_COUNT){
        throw invalid_argument("Error: mutation_type_probabilities_ must have " + to_string(MUTATION_TYPES_COUNT) + " elements.");
    }
    mutation_type_probabilities_sum = 0;
    for (uint i = 0; i < mutation_type_probabilities.size(); i++){
        mutation_type_probabilities_sum += mutation_type_probabilities[i];
    }
};

vector<nnfloat> Network::run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat)){
    // Initialize neuron values
    for (Neuron* n : neurons) {
        n->temp_value = 0;
    }

    // Set input neuron values
    for (uint input_neuron_i = 0; input_neurons.size() > input_neuron_i; input_neuron_i++){
        neurons[input_neurons[input_neuron_i]] -> temp_value = input[input_neuron_i];
    }

    for (uint step = 0; steps > step; step++){
        // Use temp_value2 to store the next state's accumulated values
        for (Neuron* n : neurons) {
            n->temp_value2 = 0;
        }

        // Accumulate weighted inputs
        for (Connection* c : connections) {
            c->target->temp_value2 += c->source->temp_value * c->weight;
        }

        // Apply bias and activation function, then update the main temp_value
        for (Neuron* n : neurons) {
            n->temp_value = nomalize(n->temp_value2 + n->bias);
        }
    }

    vector<nnfloat> output = vector<nnfloat>();
    for (uint output_neuron_i = 0; output_neurons.size() > output_neuron_i; output_neuron_i++){
        // The final value is in temp_value
        output.push_back(neurons[output_neurons[output_neuron_i]]->temp_value);
    }
    return output;
};
vector<nnfloat> Network::run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps){
    return run(input_neurons, input, output_neurons, steps, sigmoid);
};
vector<nnfloat> Network::run(vector<nnfloat> input, uint steps){
    return run(default_input_neurons, input, default_output_neurons, steps, sigmoid);
}
vector<nnfloat> Network::run(vector<nnfloat> input, uint steps, nnfloat normalize(nnfloat)){
    return run(default_input_neurons, input, default_output_neurons, steps, normalize);
}
Network Network::copy(){
    vector<Neuron*> new_neurons;
    unordered_map<Neuron*, Neuron*> old_to_new_neuron_map;
    for (Neuron* n : neurons) {
        Neuron* new_n = new Neuron(n->bias);
        new_neurons.push_back(new_n);
        old_to_new_neuron_map[n] = new_n;
    }

    vector<Connection*> new_connections;
    for (Connection* c : connections) {
        Neuron* new_source = old_to_new_neuron_map[c->source];
        Neuron* new_target = old_to_new_neuron_map[c->target];
        Connection* new_c = new Connection(new_source, new_target, c->weight);
        new_connections.push_back(new_c);
    }
    return Network(new_neurons, new_connections, default_input_neurons, default_output_neurons, mutations_per_copy_range, mutations_size_range, mutation_type_probabilities);

}

// void Network::save_to_file(string filename, bool overwrite){
    
// }
