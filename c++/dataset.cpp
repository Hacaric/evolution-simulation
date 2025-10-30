#include "neurons.h"

// class Dataset{
//     Dataset();
//     nnfloat getCost(Network nn, uint steps, uint sample_start, uint sample_end);
//     nnfloat getCost(Network nn, uint steps, uint sample_lenght);
//     nnfloat getCostRandomSample(Network nn, uint steps, uint sample_lenght);
// };

Dataset::Dataset(uint input_size_, uint output_size_){
    cout << "Unfinished feature #TODO" << endl;
    // throw ("Dataset is unfinished feature. #TODO");
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
}

nnfloat Dataset::getCost(Network nn, uint steps, uint sample_start, uint sample_end){
    if (sample_start >= input_size || sample_end >= input_size){
        // cout << "Error: Sample start or end index out of bounds for dataset." << endl;
        throw out_of_range("Error: Sample start or end index out of bounds for dataset.");
    }

}
