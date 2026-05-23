#pragma once
#include "layers.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

struct EmbeddingConfig {
    size_t input_col_index;
    size_t vocab_size;
    size_t embed_dim;
};

class TabularEmbeddingLayer : public Layer {
private:
    std::vector<EmbeddingConfig> configs;
    std::vector<Matrix> embedding_tables;
    std::vector<Matrix> embedding_gradients;

public:
	TabularEmbeddingLayer(size_t id, size_t input_cols, const std::vector<EmbeddingConfig>& confs); //Initialize with configs
	TabularEmbeddingLayer(size_t id, size_t input_cols, size_t output_cols);//Initialize without configs, for loading a model

    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;

    bool initialize() override;
    void update_params(double lr, size_t batch_size) override;
    void zero_gradients() override;

    std::string get_layer_name() const override { return "TabularEmbedding"; }

    void save(std::ostream& out) const override; 
    void load(std::istream& in) override; 
};