#pragma once
#include <vector>
#include <unordered_map>
using namespace std;

typedef float nnfloat;
typedef unsigned uint;

nnfloat sigmoid(nnfloat x);



struct Neuron{
    nnfloat temp_value; nnfloat temp_value2;
    vector<Neuron*> connections;
    Neuron(){
        connections = vector<Neuron*>();
    };
};

struct Connection{
    Neuron* source;
    Neuron* target;
    nnfloat weight;
    Connection(Neuron* neuron1_, Neuron* neuron2_, nnfloat weight_);
};

class Network{
private:
    vector<Neuron*> neurons;
    vector<Connection*> connections;
    vector<vector<Connection*>> map;
    unordered_map<Neuron*, uint> map_neuron_to_index; 
public:
    Network(vector<Neuron*>, vector<Connection*>);
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps);
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat));
};
