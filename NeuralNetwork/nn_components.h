#ifndef NN_COMPONENTS_H
#define NN_COMPONENTS_H

#include <fstream>
#include <vector>
#include "data_structures.h"

class Layer {
protected:
    size_t layer_id;
    Matrix accumulated_gradients; 

public:
    Layer(size_t id) : layer_id(id) {};
    virtual ~Layer() = default; 

    virtual Matrix feedforward(const Matrix& inputs) = 0;
    virtual Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) = 0;

    virtual bool initialize() = 0;
    virtual void update_params(double lr, size_t batch_size) = 0;

    virtual void zero_gradients();
    virtual bool save(std::ofstream& out) const = 0;
    virtual bool load(std::ifstream& in) = 0;

    size_t get_id() const { return layer_id; }
};

class Dense : public Layer {
private:
    Matrix weights;
    Matrix bias;

public:
    Dense(size_t id, size_t in_dim, size_t out_dim);
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    bool initialize() override;
    void update_params(double lr, size_t batch_size) override;
    bool save(std::ofstream& out) const override;
    bool load(std::ifstream& in) override;
};

class Dropout : public Layer {
private:
    float dropout_rate;
    Matrix mask; 

public:
    Dropout(size_t id, float rate);
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    bool initialize() override { return true; } 
    void update_params(double lr, size_t batch_size) override {}
    bool save(std::ofstream& out) const override;
    bool load(std::ifstream& in) override;
};

class Embedding : public Layer {
private:
    DatasetMetadata metadata;
    Matrix weights;

public:
    Embedding(size_t id, const DatasetMetadata& meta);
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    bool initialize() override;
    void update_params(double lr, size_t batch_size) override;
    bool save(std::ofstream& out) const override;
    bool load(std::ifstream& in) override;
};

class ReLU : public Layer {
public:
    ReLU(size_t id) : Layer(id) {}
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    bool initialize() override { return true; }
    void update_params(double lr, size_t batch_size) override {}
    bool save(std::ofstream& out) const override { return true; }
    bool load(std::ifstream& in) override { return true; }
};

class Sigmoid : public Layer {
public:
    Sigmoid(size_t id) : Layer(id) {}
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    bool initialize() override { return true; }
    void update_params(double lr, size_t batch_size) override {}
    bool save(std::ofstream& out) const override { return true; }
    bool load(std::ifstream& in) override { return true; }
};

#endif