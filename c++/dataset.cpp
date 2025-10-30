#include "neurons.h"
#include <string>

// class Dataset{
//     Dataset();
//     nnfloat getNetworkCost(Network nn, uint steps, uint sample_start, uint sample_end);
//     nnfloat getNetworkCost(Network nn, uint steps, uint sample_lenght);
//     nnfloat getCostRandomSample(Network nn, uint steps, uint sample_lenght);
// };

Dataset::Dataset(uint input_size_, uint output_size_){
    input_size = input_size_;
    output_size = output_size_;
}

void Dataset::loadData_ByReference(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> labels_){
    inputs = inputs_;
    labels = labels_;
}

void Dataset::loadData_ByCopying(vector<vector<nnfloat>> inputs_, vector<vector<nnfloat>> labels_){
    inputs = vector<vector<nnfloat>>(inputs_);
    labels = vector<vector<nnfloat>>(labels_);
    this->loadData_ByReference(inputs, labels);
}

void Dataset::addEntry_ByReference(vector<nnfloat> input_, vector<nnfloat> label_){
    if (input_.size() != input_size){
        throw invalid_argument("Error: Input vector size does not match dataset's input_size. Required: " + to_string(input_size) + ", got: " + to_string(input_.size()));
    }
    if (label_.size() != output_size){
        throw invalid_argument("Error: Label vector size does not match dataset's output_size. Required: " + to_string(output_size) + ", got: " + to_string(label_.size()));
    }
    inputs.push_back(input_);
    labels.push_back(label_);
}
void Dataset::addEntry_ByCopying(vector<nnfloat> input_, vector<nnfloat> label_){
    vector<nnfloat> input = vector<nnfloat>(input_);
    vector<nnfloat> label = vector<nnfloat>(label_);
    this->addEntry_ByReference(input, label);
}


nnfloat CostFunc(Network nn, vector<nnfloat> input, vector<nnfloat> label, uint steps){
    vector<nnfloat> output = nn.run(input, steps);
    cout << "nn.default_output_neurons.size(): " << nn.default_output_neurons.size() << ", label.size()" << label.size() << endl;
    if (output.size() != label.size()){
        throw runtime_error("Error in dataset.cpp-CostFunc: Network output size does not match label size.");
    }
    nnfloat cost = 0;
    for (uint output_node_i = 0; output_node_i < output.size(); output_node_i++){
        cost += (label[output_node_i] - output[output_node_i]) * (label[output_node_i] - output[output_node_i]);
    }
    return cost;
}
// Calculates cost for neural network based on sample from dataset
nnfloat Dataset::getNetworkCost(Network nn, uint steps, uint sample_start, uint sample_end){
    if (sample_start >= inputs.size() || sample_end >= inputs.size()){
        // cout << "Error: Sample start or end index out of bounds for dataset." << endl;
        throw out_of_range("Error: Sample start or end index out of bounds for dataset.");
    }
    if (sample_start >= sample_end){
        throw invalid_argument("Error: Sample start index must be less than sample end index.");
    }

    nnfloat cost = 0;
    for (uint label_i = sample_start; label_i < sample_end; label_i++){
        cost += CostFunc(nn, inputs[label_i], labels[label_i], steps);
    }

    return cost;
}
nnfloat Dataset::getNetworkCost(Network nn, uint steps, uint sample_lenght){
    return this->getNetworkCost(nn, steps, 0, sample_lenght);
}

nnfloat Dataset::getNetworkCost(Network nn, uint steps){
    return this->getNetworkCost(nn, steps, 0, this->inputs.size()-1);
}
