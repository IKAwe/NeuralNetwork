#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include "data_structures.h"
#include "data_preprocessor.h"
#include "nn_components.h"
#include "loss_functions.h"

struct Hyperparams {
    int epochs=10;
    int batch_size=32;
    float learning_rate=0.01;
	Hyperparams() = default;
    Hyperparams(int e, int b, float lr) : epochs(e), batch_size(b), learning_rate(lr) {}
};

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Loss> loss_function;

public:
    NeuralNetwork() = default;

    void add_layer(std::unique_ptr<Layer> layer);
    void delete_last_layer();

    void clear_layers() {layers.clear();}

    Matrix predict(const Matrix& inputs);

    void train(const Dataset& dataset, const Hyperparams params);

    void set_loss(std::unique_ptr<Loss> loss) {
        loss_function = std::move(loss);
	}
    double test(const Matrix& inputs, const Matrix& targets);

    bool save(const std::string& filename);
    bool load(const std::string& filename);
};

#endif