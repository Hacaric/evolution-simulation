#pragma once
#include <vector>
#include <unordered_map>
#include <iostream>
#include <math.h>
#include <string>

using namespace std;

typedef float nnfloat;
#define MAXNNFLOAT MAXFLOAT
#define MUTATION_TYPES_COUNT 6
typedef unsigned uint;

nnfloat sigmoid(nnfloat x);


struct Neuron{
    nnfloat bias;
    nnfloat temp_value; nnfloat temp_value2;
    vector<Neuron*> connections;
    Neuron(nnfloat bias_){
        bias = bias_;
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
public:
    nnfloat mutation_type_probabilities_sum;
    pair<uint, uint> mutations_per_copy_range;
    pair<nnfloat, nnfloat> mutations_size_range;
    vector<uint> default_input_neurons, default_output_neurons;
    vector<nnfloat> mutation_type_probabilities;
    vector<Neuron*> neurons;
    vector<Connection*> connections;
    Network(vector<Neuron*> neurons, vector<Connection*> connections, vector<uint> default_input_neurons_, vector<uint> default_output_neurons_, pair<uint, uint> mutations_per_copy_range_, pair<nnfloat, nnfloat> mutation_size_range_, vector<nnfloat> mutation_type_probabilities);
    void save_to_file(string filename, bool overwrite);
    vector<nnfloat> run(vector<nnfloat> input, uint steps);
    vector<nnfloat> run(vector<nnfloat> input, uint steps, nnfloat nomalize(nnfloat));
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps);
    vector<nnfloat> run(vector<uint> input_neurons, vector<nnfloat> input, vector<uint> output_neurons, uint steps, nnfloat nomalize(nnfloat));
    Network copy();
};


class Dataset{
public:
    uint input_size, output_size;
    vector<vector<nnfloat>> inputs, labels;
    
    Dataset(uint input_size_, uint output_size_);
    void loadData_ByReference(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> lables_);
    void loadData_ByCopying(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> labels_);
    void addEntry_ByReference(vector<nnfloat> input_, vector<nnfloat> label_);
    void addEntry_ByCopying(vector<nnfloat> input_, vector<nnfloat> label_);
    nnfloat getNetworkCost(Network& nn, uint steps, uint sample_start, uint sample_end);
    nnfloat getNetworkCost(Network& nn, uint steps, uint sample_lenght);
    nnfloat getNetworkCost(Network& nn, uint steps);
};
