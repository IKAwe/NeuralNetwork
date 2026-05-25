#include <cmath>
#include "layers.h"

class Sigmoid : public Layer {
public:
    Sigmoid(size_t id, size_t size) : Layer(id, size, size) {}

    bool initialize() override { return true; }
    void update_params(double learning_rate, size_t batch_size) override {}

    Matrix feedforward(const Matrix& inputs) override {
        Matrix result(inputs.get_rows_nb(), inputs.get_columns_nb());

        const auto& in_data = inputs.get_data();
        auto& out_data = result.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            double x = std::max(-40.0, std::min(40.0, in_data[i]));
            out_data[i] = 1.0 / (1.0 + std::exp(-x));
        }
        return result;
    }

    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override {
        Matrix derivative(inputs.get_rows_nb(), inputs.get_columns_nb());

        const auto& in_data = inputs.get_data();
        auto& deriv_data = derivative.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            double x = std::max(-40.0, std::min(40.0, in_data[i]));
            double sig = 1.0 / (1.0 + std::exp(-x));
            deriv_data[i] = sig * (1.0 - sig);
        }

        return gradients_from_next_layer.hadamard(derivative);
    }

    void zero_gradients() override {};
    std::string get_layer_name() const override { return "Sigmoid"; };


};

class ReLU : public Layer {
    public:
    ReLU(size_t id, size_t size) : Layer(id, size, size) {}
    bool initialize() override { return true; }
    void update_params(double learning_rate, size_t batch_size) override {}
    Matrix feedforward(const Matrix& inputs) override {
        Matrix result(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& out_data = result.get_data_mutable();
        for (size_t i = 0; i < in_data.size(); ++i) {
            out_data[i] = std::max(0.0, in_data[i]);
        }
        return result;
    }
    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override {
        Matrix derivative(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& deriv_data = derivative.get_data_mutable();
        for (size_t i = 0; i < in_data.size(); ++i) {
            deriv_data[i] = in_data[i] > 0.0 ? 1.0 : 0.0;
        }
        return gradients_from_next_layer.hadamard(derivative);
    }
    
	void zero_gradients() override {};
	std::string get_layer_name() const override { return "ReLU"; }
};


class Tanh : public Layer {
public:
    Tanh(size_t id, size_t size) : Layer(id, size, size) {}

    bool initialize() override { return true; }
    void update_params(double learning_rate, size_t batch_size) override {}
    void zero_gradients() override {}
    std::string get_layer_name() const override { return "Tanh"; }

    Matrix feedforward(const Matrix& inputs) override {
        Matrix result(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& out_data = result.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            out_data[i] = std::tanh(in_data[i]);
        }
        return result;
    }

    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override {
        Matrix derivative(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& deriv_data = derivative.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            // tanh'(x) = 1 - tanh^2(x)
            double t = std::tanh(in_data[i]);
            deriv_data[i] = 1.0 - (t * t);
        }
        return gradients_from_next_layer.hadamard(derivative);
    }
};

class LeakyReLU : public Layer {
private:
    double alpha; // Leak parameter 0.01

public:
    LeakyReLU(size_t id, size_t size, double leak_alpha = 0.01)
        : Layer(id, size, size), alpha(leak_alpha) {
    }

    bool initialize() override { return true; }
    void update_params(double learning_rate, size_t batch_size) override {}
    void zero_gradients() override {}
    std::string get_layer_name() const override { return "LeakyReLU"; }

    Matrix feedforward(const Matrix& inputs) override {
        Matrix result(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& out_data = result.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            out_data[i] = in_data[i] > 0.0 ? in_data[i] : alpha * in_data[i];
        }
        return result;
    }

    Matrix backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) override {
        Matrix derivative(inputs.get_rows_nb(), inputs.get_columns_nb());
        const auto& in_data = inputs.get_data();
        auto& deriv_data = derivative.get_data_mutable();

        for (size_t i = 0; i < in_data.size(); ++i) {
            // Pochodna: 1.0 jeœli wejœcie > 0, alpha w przeciwnym razie
            deriv_data[i] = in_data[i] > 0.0 ? 1.0 : alpha;
        }
        return gradients_from_next_layer.hadamard(derivative);
    }
};