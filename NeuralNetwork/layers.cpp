#include "layers.h"
#include <random>

Dense::Dense(size_t id, size_t in_dim, size_t out_dim)
    : Layer(id, in_dim, out_dim), weights(in_dim, out_dim), bias(1, out_dim), accumulated_gradients(in_dim, out_dim), accumulated_bias_gradients(1, out_dim) {}

bool Dense::initialize() {
    std::random_device rd;
    std::mt19937 generator(rd());

    double std_dev = 1.0;
    double n_in = static_cast<double>(weights.get_rows_nb());
    double n_out = static_cast<double>(weights.get_columns_nb());

    if (init_method == InitializationMethod::He) {
        // He: ReLU
        std_dev = std::sqrt(2.0 / n_in);
    }
    else if (init_method == InitializationMethod::Xavier) {
        // Xavier - Sigmoid/Tanh/Linear
        std_dev = std::sqrt(2.0 / (n_in + n_out));
    }

    std::normal_distribution<double> distribution(0.0, std_dev);

    for (size_t r = 0; r < weights.get_rows_nb(); ++r) {
        for (size_t c = 0; c < weights.get_columns_nb(); ++c) {
            weights(r, c) = distribution(generator);
        }
    }
    for (size_t c = 0; c < bias.get_columns_nb(); ++c) {
        bias(0, c) = 0.0; // Bias initialization with 0's
    }
    return true;
}

Matrix Dense::feedforward(const Matrix& inputs) {
    // Y= X * W + B
    Matrix output = inputs * weights;
    // Add biases
    for (size_t r = 0; r < output.get_rows_nb(); ++r) {
        for (size_t c = 0; c < output.get_columns_nb(); ++c) {
            output(r, c) += bias(0, c);
        }
    }
    return output;
}

Matrix Dense::backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) {
    // Calculate gradients(dL/dw)
    Matrix inputs_T = inputs.transpose();
    Matrix dW = inputs_T * gradients_from_next_layer;
    // Add gradients (sum then will be divided by batch size)
    for (size_t r = 0; r < dW.get_rows_nb(); ++r) {
        for (size_t c = 0; c < dW.get_columns_nb(); ++c) {
            accumulated_gradients(r, c) += dW(r, c);
        }
    }
	// Calculate gradients for bias (dL/db)
    for (size_t r = 0; r < gradients_from_next_layer.get_rows_nb(); ++r) {
        for (size_t c = 0; c < gradients_from_next_layer.get_columns_nb(); ++c) {
            // Bias to wektor wierszowy (1 x C), wiêc indeks wiersza to zawsze 0
            accumulated_bias_gradients(0, c) += gradients_from_next_layer(r, c);
        }
    }
    // Calc graident for the inputs to the layer (dL/dz)
    Matrix weights_T = weights.transpose();
    Matrix gradients_to_prev_layer = gradients_from_next_layer * weights_T;

    return gradients_to_prev_layer;
}

void Dense::update_params(double lr, size_t batch_size) {

    // Update weigths: W = W -(lr/batch_size)* dW
    weights = weights - (accumulated_gradients * lr);
    bias = bias - (accumulated_bias_gradients * lr);
    // Zero the gradients after a batch
    zero_gradients();
}


void Dense::save(std::ostream& out) const {
    weights.save(out);
    bias.save(out);
}

void Dense::load(std::istream& in) {
    weights.load(in);
    bias.load(in);
}