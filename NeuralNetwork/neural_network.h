#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include "data_structures.h"
#include "data_preprocessor.h"
#include "nn_components.h"
#include "data_structures.h"

struct Hyperparams {
    size_t epochs;
    size_t batch_size;
    double learning_rate;
    Hyperparams(size_t e, size_t b, double lr) : epochs(e), batch_size(b), learning_rate(lr) {}
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    //std::unique_ptr<Loss> loss_function;

public:
    NeuralNetwork() = default;

    void add_layer(std::unique_ptr<Layer> layer);
    void delete_last_layer();

    //void set_loss(std::unique_ptr<Loss> loss);

    Matrix predict(const Matrix& inputs);

    void train(const Dataset& dataset, const Hyperparams& params);

    Matrix test(const Matrix& inputs, const Matrix& targets);

    bool save(const std::string& filename);
    bool load(const std::string& filename);
};

#endif