#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <memory>
#include <string>
#include "data_structures.h"
#include "data_preprocessor.h"
#include "layers.h"
#include "loss_functions.h"
#include <functional>

struct Hyperparams {
    int epochs=10;
    int batch_size=32;
    float learning_rate=0.01;
	Hyperparams() = default;
    Hyperparams(int e, int b, float lr) : epochs(e), batch_size(b), learning_rate(lr) {}
};
struct EpochStats {
    int epoch;
    double loss;
    EpochStats(int e, double l) : epoch(e), loss(l) {}
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

    void train(const Dataset& dataset, const Hyperparams params, std::function<void(EpochStats)> on_epoch_end = nullptr);

    void set_loss(std::unique_ptr<Loss> loss) {
        loss_function = std::move(loss);
	}
    double test(const Matrix& inputs, const Matrix& targets);

    void save(const std::string& filename);
    void load(const std::string& filename);

	//For GUI when loading a model, to display the architecture
	const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers; }
};

#endif