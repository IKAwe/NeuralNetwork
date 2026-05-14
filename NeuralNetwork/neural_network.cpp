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

    double error = loss_function->calculate(predictions, targets);

    return error;
}


void NeuralNetwork::train(const Dataset& dataset, const Hyperparams params, std::function<void(EpochStats)> on_epoch_end) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::train - Loss function was not added");
    }

    const Matrix& inputs = dataset.input_data;
    const Matrix& targets = dataset.output_data;
    size_t total_samples = inputs.get_rows_nb();
    
    if (total_samples == 0) {
        throw std::runtime_error("NeuralNetwork::train - Dataset is empty");
    }
    if (targets.get_columns_nb() == 0) {
        throw std::runtime_error("NeuralNetwork:train - No target column");
    }
    if (inputs.get_columns_nb() != layers.front()->get_input_nb()) {
        throw std::runtime_error(
            "NeuralNetwork::train - Input dimension mismatch. "
            "Dataset input columns: " + std::to_string(inputs.get_columns_nb()) +
            ", Expected: " + std::to_string(layers.front()->get_input_nb())
        );
	}
    if (targets.get_columns_nb() != layers.back()->get_output_nb()) {
        throw std::runtime_error(
            "NeuralNetwork::train - Target dimension mismatch. "
            "Dataset target columns: " + std::to_string(targets.get_columns_nb()) +
            ", Expected: " + std::to_string(layers.back()->get_output_nb())
        );
	}
    //Preallocation for batching
    Matrix batch_inputs(params.batch_size, inputs.get_columns_nb());
    Matrix batch_targets(params.batch_size, targets.get_columns_nb());

    for (size_t epoch = 0; epoch < params.epochs; ++epoch) {

		double accumulated_loss = 0.0;
		int batch_count = 0;

        for (size_t start_idx = 0; start_idx < total_samples; start_idx += params.batch_size) {

            size_t end_idx = std::min(start_idx + params.batch_size, total_samples);
            size_t current_batch_size = end_idx - start_idx;

            batch_inputs.overwrite_with_rows(inputs, start_idx, end_idx);
            batch_targets.overwrite_with_rows(targets, start_idx, end_idx);

            // --- 1. FORWARD PASS ---
            std::vector<Matrix> layer_inputs;
            layer_inputs.push_back(batch_inputs);

            Matrix current_data = batch_inputs;
            for (const auto& layer : layers) {
                current_data = layer->feedforward(current_data);
                layer_inputs.push_back(current_data);
            }
            Matrix predictions = current_data;

            accumulated_loss += loss_function->calculate(predictions, batch_targets);
            batch_count++;

            // --- 2. BACKWARD PASS ---
            Matrix current_gradient = loss_function->calculate_gradient(predictions, batch_targets);

            for (int l = (int)layers.size() - 1; l >= 0; --l) {
                current_gradient = layers[l]->backpropagate(layer_inputs[l], current_gradient);
            }

            // --- 3. WEIGTH UPDATE ---
            for (auto& layer : layers) {
                layer->update_params(params.learning_rate, current_batch_size);
            }
        }
        double epoch_loss = accumulated_loss / batch_count;

        size_t print_interval = (params.epochs >= 10) ? (params.epochs / 10) : 1;
        if (epoch % print_interval == 0 || epoch == params.epochs - 1) {
            std::cout << "Epoch [" << epoch + 1 << "/" << params.epochs << "] - Loss: " << epoch_loss << std::endl;
        }
		//Send epoch stats to callback
        if (on_epoch_end) {
            on_epoch_end(EpochStats(epoch+1, epoch_loss));
		}
    }
}