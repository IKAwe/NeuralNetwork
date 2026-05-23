#ifndef NN_COMPONENTS_H
#define NN_COMPONENTS_H

#include <fstream>
#include <vector>
#include "data_structures.h"

enum class InitializationMethod {
	Xavier, //Best for tanh/sigmoid
	He //Best for ReLU and its variants
};

class Layer {
protected:
    size_t input_nb = 0;
    size_t output_nb = 0;
    size_t layer_id;

public:
    Layer(size_t id) : layer_id(id) {};
    Layer(size_t id, size_t inputs_nb_passed, size_t outputs_nb_passed) : layer_id(id), input_nb(inputs_nb_passed), output_nb(outputs_nb_passed){};
    virtual ~Layer() = default; 

    virtual Matrix feedforward(const Matrix& inputs) = 0;
    virtual Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) = 0;

    virtual bool initialize() = 0;
    virtual void update_params(double lr, size_t batch_size) = 0;

    virtual void zero_gradients()=0;

    // Storing/loading
    virtual std::string get_layer_name() const = 0;
    virtual void save(std::ostream& out) const {};
    virtual void load(std::istream& in) {};

    size_t get_id() const { return layer_id; }

	size_t get_input_nb() const { return input_nb; }
	size_t get_output_nb() const { return output_nb; }
};

class Dense : public Layer {
private:
    Matrix accumulated_gradients;
	Matrix accumulated_bias_gradients;
    Matrix weights;
    Matrix bias;
    InitializationMethod init_method = InitializationMethod::He;

public:
    Dense(size_t id, size_t in_dim, size_t out_dim);
    Matrix feedforward(const Matrix& inputs) override;
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
    void set_initialization_method(InitializationMethod method) { init_method = method; };
    bool initialize() override;
    void update_params(double lr, size_t batch_size) override;
	std::string get_layer_name() const override { return "Dense"; }
    void save(std::ostream& out) const override;
    void load(std::istream& in) override;

    void zero_gradients() override {accumulated_gradients.zero(); accumulated_bias_gradients.zero(); } ;

#ifdef _DEBUG
    // For testing purposes
    Matrix& get_weights() { return weights; }
	Matrix& get_bias() { return bias; }
	Matrix& get_accumulated_gradients() { return accumulated_gradients; }
#endif


};

//class Dropout : public Layer {
//private:
//    float dropout_rate;
//    Matrix mask; 
//
//public:
//    Dropout(size_t id, float rate);
//    Matrix feedforward(const Matrix& inputs) override;
//    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override;
//    bool initialize() override { return true; } 
//    void update_params(double lr, size_t batch_size) override {}
//    bool save(std::ofstream& out) const override;
//    bool load(std::ifstream& in) override;
//};


#endif