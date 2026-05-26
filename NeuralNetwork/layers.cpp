#include "layers.h"
#include <random>

Dense::Dense(size_t id, size_t in_dim, size_t out_dim)
    : Layer(id, in_dim, out_dim), weights(in_dim, out_dim), bias(1, out_dim), accumulated_gradients(in_dim, out_dim), accumulated_bias_gradients(1, out_dim) {}

/**
 * @brief Initializes weights and biases of the layer using the specified initialization method (He or Xavier).
 * @return True if initialization is successful, false otherwise.
 */
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
/**
 * @brief Feedforward pass for the dense layer. It computes the output as Y = X * W + B, where X is the input matrix, W are the weights, and B are the biases.
 * @param inputs The input matrix to the layer, with dimensions (batch_size x input_dim).
 * @return The output matrix of the layer, with dimensions (batch_size x output_dim).
 */
Matrix Dense::feedforward(const Matrix& inputs) {
    // Y= X * W + B
    Matrix output = inputs * weights;
    size_t rows = output.get_rows_nb();
    size_t cols = output.get_columns_nb();
    // Add biases
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            output(r, c) += bias(0, c);
        }
    }
    return output;
}
/**
 * @brief Applies backpropagation for the dense layer. It calculates the gradients with respect to weights and biases, accumulates them, and returns the gradients with respect to the inputs for the previous layer.
 * @param inputs the input matrix that was fed into the layer during the feedforward pass, with dimensions (batch_size x input_dim).
 * @param gradients_from_next_layer The gradients received from the next layer in the network, with dimensions (batch_size x output_dim).
 * @return The gradients with respect to the inputs of this layer, which will be passed to the previous layer during backpropagation, with dimensions (batch_size x input_dim).
 */
Matrix Dense::backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) {
    // Calculate gradients(dL/dw)
    Matrix inputs_T = inputs.transpose();
    Matrix dW = inputs_T * gradients_from_next_layer;
    // Add gradients (sum then will be divided by batch size)
    accumulated_gradients += dW;
	// Calculate gradients for bias (dL/db)
    size_t grad_rows = gradients_from_next_layer.get_rows_nb();
    size_t grad_cols = gradients_from_next_layer.get_columns_nb();

    for (size_t r = 0; r < grad_rows; ++r) {
        for (size_t c = 0; c < grad_cols; ++c) {
            // Bias to wektor wierszowy (1 x C), wiêc indeks wiersza to zawsze 0
            accumulated_bias_gradients(0, c) += gradients_from_next_layer(r, c);
        }
    }
    // Calc graident for the inputs to the layer (dL/dz)
    Matrix weights_T = weights.transpose();
    return gradients_from_next_layer * weights_T;
}

//Update parameters - we assume that Loss funciton already divided gradients by batch size.
/**@brief Updates the weights and biases of the dense layer using the accumulated gradients and the specified learning rate. After updating, it resets the accumulated gradients to zero for the next batch.
 * @param lr The learning rate to be used for updating the parameters.
 * @param batch_size The size of the batch that was used to accumulate the gradients (not used in this implementation since we assume gradients are already averaged).
**/
void Dense::update_params(double lr, size_t batch_size) {
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