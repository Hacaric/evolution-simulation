#pragma once
#include <vector>
#include <unordered_map>
#include <iostream>

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
    vector<vector<Connection*>> map;
    unordered_map<Neuron*, uint> map_neuron_to_index; 
public:
    vector<uint> default_input_neurons, default_output_neurons;
    
    vector<Neuron*> neurons;
    vector<Connection*> connections;
    Network(vector<Neuron*> neurons, vector<Connection*> connections, vector<uint> default_input_neurons_, vector<uint> default_output_neurons_);
    vector<nnfloat> run(vector<nnfloat> input, uint steps);
    vector<nnfloat> run(vector<nnfloat> input, uint steps, nnfloat nomalize(nnfloat));
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps);
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat));
    Network copy();
};


class Dataset{
private:
    uint input_size, output_size;
    vector<vector<nnfloat>> inputs, labels;
    
public:
    Dataset(uint input_size_, uint output_size_);
    void loadData_ByReference(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> lables_);
    void loadData_ByCopying(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> labels_);
    nnfloat getCost(Network nn, uint steps, uint sample_start, uint sample_end);
    nnfloat getCost(Network nn, uint steps, uint sample_lenght);
    nnfloat getCostRandomSample(Network nn, uint steps, uint sample_lenght);
};
