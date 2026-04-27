#include "neural_network.h"
#include "loss_functions.h"
#include <iostream>
/**
 * @brief Add previosly initialized layer to the model.
 * @param layer Unique pointer to the layer to be added. The function takes ownership of the layer and will manage its lifetime. The layer should be initialized before being added to the model.
 */
void NeuralNetwork::add_layer(std::unique_ptr<Layer> layer) {
    if (layer == nullptr) {
        throw std::invalid_argument("NeuralNetwork::add_layer - cannot add a null layer.");
    }
    //Check size
    if (!layers.empty()) {
        size_t prev_output = layers.back()->get_output_nb();
        size_t new_input = layer->get_input_nb();

        if (prev_output != new_input) {
            throw std::runtime_error(
                "NeuralNetwork::add_layer - Dimension mismatch. "
                "Previous layer output: " + std::to_string(prev_output) +
                ", New layer input: " + std::to_string(new_input)
            );
        }
    }
    ////Initialize
    //if (!layer->initialize()) {
    //    throw std::runtime_error("NeuralNetwork::add_layer - Layer initialization failed.");
    //}
    layers.push_back(std::move(layer));
}

void NeuralNetwork::delete_last_layer() {
    if (layers.empty()) {
        return;
    }
    layers.pop_back();
}

Matrix NeuralNetwork::predict(const Matrix& inputs) {
    if (layers.empty()) {
        throw std::runtime_error("NeuralNetwork::predict - Model has no layers.");
    }
    Matrix current_data = inputs;
    for (const auto& layer : layers) {
        current_data = layer->feedforward(current_data);
    }
    return current_data;
}


double NeuralNetwork::test(const Matrix& inputs, const Matrix& targets) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::test - Nie ustawiono funkcji straty!");
    }

    Matrix predictions = predict(inputs);

    // U¿ywamy naszej nowej klasy
    double error = loss_function->calculate(predictions, targets);

    return error;
}


void NeuralNetwork::train(const Dataset& dataset, const Hyperparams& params) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::train - Loss function was not added");
    }

    const Matrix& inputs = dataset.input_data;
    const Matrix& targets = dataset.output_data;
    size_t total_samples = inputs.get_rows_nb();

    if (total_samples == 0) {
        throw std::runtime_error("NeuralNetwork::train - Dataset is empty");
    }

    for (size_t epoch = 0; epoch < params.epochs; ++epoch) {

        std::vector<Matrix> layer_inputs;
        layer_inputs.push_back(inputs); 

        Matrix current_data = inputs;
        for (const auto& layer : layers) {
            current_data = layer->feedforward(current_data);
            layer_inputs.push_back(current_data);
        }

        Matrix predictions = current_data;

        size_t print_interval = (params.epochs >= 10) ? (params.epochs / 10) : 1;
        if (epoch % print_interval == 0 || epoch == params.epochs - 1) {
            double current_loss = loss_function->calculate(predictions, targets);
            std::cout << "Epoch [" << epoch << "/" << params.epochs << "] - Loss: " << current_loss << std::endl;
        }

        Matrix current_gradient = loss_function->calculate_gradient(predictions, targets);

        for (int l = layers.size() - 1; l >= 0; --l) {
            current_gradient = layers[l]->backpropagate(layer_inputs[l], current_gradient);
        }

        for (auto& layer : layers) {
            layer->update_params(params.learning_rate, total_samples);
        }
    }
}