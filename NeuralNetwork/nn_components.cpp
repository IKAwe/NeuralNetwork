#include "nn_components.h"
#include <random>

Dense::Dense(size_t id, size_t in_dim, size_t out_dim)
    : Layer(id, in_dim, out_dim), weights(in_dim, out_dim), bias(1, out_dim), accumulated_gradients(in_dim, out_dim), accumulated_bias_gradients(1, out_dim) {}

bool Dense::initialize() {
    // He initialization
    std::default_random_engine generator;
    double std_dev = std::sqrt(2.0 / weights.get_rows_nb());
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
    double factor= lr/static_cast<double>(batch_size);

    // Update weigths: W = W -(lr/batch_size)* dW
    for (size_t r = 0; r < weights.get_rows_nb(); ++r) {
        for (size_t c = 0; c < weights.get_columns_nb(); ++c) {
            weights(r, c) -= factor * accumulated_gradients(r, c);
        }
    }
    //Update bias
    for (size_t r = 0; r < bias.get_rows_nb(); ++r) {
        for (size_t c = 0; c < bias.get_columns_nb(); ++c) {
            // Zak³adam, ¿e masz zmienn¹ przetrzymuj¹c¹ gradienty biasu z backpropagate!
            bias(r, c) -= factor * accumulated_bias_gradients(r, c);
        }
    }
    // Zero the gradients after a batch
    zero_gradients();
}

bool Dense::save(std::ofstream& output_fstream) const {
    if (!output_fstream.is_open()) return false;

    // Store dimesions and then data
    size_t rows = weights.get_rows_nb();
    size_t cols = weights.get_columns_nb();

    output_fstream.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    output_fstream.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
    output_fstream.write(reinterpret_cast<const char*>(weights.get_data().data()), rows * cols * sizeof(double));
    output_fstream.write(reinterpret_cast<const char*>(bias.get_data().data()), cols * sizeof(double));

    return output_fstream.good();
}

bool Dense::load(std::ifstream& input_fstream) {
    if (!input_fstream.is_open()) return false;
    size_t rows, cols;
    input_fstream.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    input_fstream.read(reinterpret_cast<char*>(&cols), sizeof(size_t));
	// alcocate memory for weights and bias
    weights = Matrix(rows, cols);
    bias = Matrix(1, cols);

    input_fstream.read(reinterpret_cast<char*>(weights.get_data_mutable().data()), rows * cols * sizeof(double));
    input_fstream.read(reinterpret_cast<char*>(bias.get_data_mutable().data()), cols * sizeof(double));

    return input_fstream.good();
}