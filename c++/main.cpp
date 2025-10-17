#include <iostream>
#include <vector>
#include "neurons.h"

using namespace std;
typedef unsigned uint;


int main(){
    vector<Neuron*> neurons = vector<Neuron*>();
    Neuron* n1 = new Neuron();
    Neuron* n2 = new Neuron();
    Neuron* n3 = new Neuron();
    neurons.push_back(n1);
    neurons.push_back(n2);
    neurons.push_back(n3);
    Connection* c1 = new Connection(neurons[0], neurons[1], nnfloat(1));
    Connection* c2 = new Connection(neurons[2], neurons[1], nnfloat(1));
    vector<Connection*> connections = vector<Connection*>();
    connections.push_back(c1);
    connections.push_back(c2);
    Network nn = Network(neurons, connections);
    vector<uint> input_neurons = vector<uint>();
    input_neurons.push_back(0);
    input_neurons.push_back(2);
    vector<nnfloat> input = vector<nnfloat>();
    input.push_back(1);
    input.push_back(0);
    vector<uint> output_neurons = vector<uint>();
    output_neurons.push_back(1);
    uint steps = 1;
    cout << "Output: " << nn.run(input_neurons, input, output_neurons, steps)[0] << endl;
    return 0;
}