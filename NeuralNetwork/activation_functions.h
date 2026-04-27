#include <cmath>
#include "nn_components.h"

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
};