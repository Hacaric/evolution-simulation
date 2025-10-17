#include <iostream>
#include <vector>
#include <unordered_map>
#include <math.h>
#include "neurons.h"

using namespace std;

nnfloat sigmoid(nnfloat x) {
    // return x;
    return 1.0f / (1.0f + exp(-x));
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

Network::Network(vector<Neuron*> neurons_, vector<Connection*> connections_){
    neurons = neurons_;
    connections = connections_;
    map = vector<vector<Connection*>>();
    for (uint neuron_i = 0; neuron_i < neurons.size(); neuron_i++){
        map.push_back(vector<Connection*>());
        map_neuron_to_index[neurons[neuron_i]] = neuron_i;
    }
    uint source_node;
    for (uint connection_i = 0; connection_i < connections.size(); connection_i++){
        source_node = map_neuron_to_index[connections[connection_i]->source];
        map[source_node].push_back(connections[connection_i]);
    }
};

vector<nnfloat> Network::run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat)){
    for (uint neuron_i = 0; neurons.size() > neuron_i; neuron_i++){
        neurons[neuron_i]->temp_value2 = 0;
    }
    for (uint input_neuron_i = 0; input_neurons.size() > input_neuron_i; input_neuron_i++){
        neurons[input_neurons[input_neuron_i]] -> temp_value = input[input_neuron_i];
    }
    Connection* curr;
    for (uint step = 0; steps > step; step++){
        // cout << "Round1: " << neurons[1]->temp_value << endl;
        // cout << "Round1: " << neurons[1]->temp_value2 << ",connsize " << connections.size() << endl;
        for (uint connection_i = 0; connections.size() > connection_i; connection_i++){
            curr = connections[connection_i];
            if (step % 2 != 0){
                // cout << curr->target->temp_value << ",(A) " << neurons[1]->temp_value << endl;
                // cout << curr->target->temp_value2 << ",(A) " << neurons[1]->temp_value2 << endl;
                curr->target->temp_value += curr->source->temp_value2 * curr->weight;
                // cout << curr->target->temp_value << ",(A2) " << neurons[1]->temp_value << endl;
                // cout << curr->target->temp_value2 << ",(A2) " << neurons[1]->temp_value2 << endl;
            }else{
                // cout << curr->target->temp_value << ",(B) " << neurons[1]->temp_value << endl;
                curr->target->temp_value2 += curr->source->temp_value * curr->weight;
                // cout << curr->target->temp_value << ",(B2) " << neurons[1]->temp_value << endl;
            }
        }
        for (uint neuron_i = 0; neurons.size() > neuron_i; neuron_i++){
            // cout << (step % 2 == 0) << "N" << neuron_i <<": " << neurons[neuron_i]->temp_value << ", " << neurons[neuron_i]->temp_value2 <<endl;
            if (step % 2 != 0){
                neurons[neuron_i]->temp_value = nomalize(neurons[neuron_i]->temp_value);
            }else{
                neurons[neuron_i]->temp_value2 = nomalize(neurons[neuron_i]->temp_value2);
            }
        }
    }
    // cout << "end: " << neurons[1]->temp_value << endl;
    // cout << "end: " << neurons[1]->temp_value2 << endl;
    vector<nnfloat> output = vector<nnfloat>();
    for (uint output_neuron_i = 0; output_neurons.size() > output_neuron_i; output_neuron_i++){
        if (steps % 2 != 0){
            output.push_back(neurons[output_neurons[output_neuron_i]]->temp_value2);
        } else {
            output.push_back(neurons[output_neurons[output_neuron_i]]->temp_value);
        }
    }
    return output;
};
vector<nnfloat> Network::run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps){
    return run(input_neurons, input, output_neurons, steps, sigmoid);
};