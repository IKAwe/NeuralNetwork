#include "neural_network.h"
#include "loss_functions.h"
#include "layer_maker.h"
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
        throw std::runtime_error("NeuralNetwork::test - No loss function was added!");
    }

    Matrix predictions = predict(inputs);

    double error = loss_function->calculate(predictions, targets);

    return error;
}


void NeuralNetwork::train(const Dataset& dataset, const Hyperparams params, std::function<bool(EpochStats)> on_epoch_end) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::train - Loss function was not added");
    }

    const Matrix& inputs = dataset.input_data;
    const Matrix& targets = dataset.output_data;
    size_t total_samples = inputs.get_rows_nb();

    const Matrix& test_inputs = dataset.test_inputs;
    const Matrix& test_targets = dataset.test_outputs;
    
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

        //========== TEST against test set ==============


        size_t print_interval = (params.epochs >= 10) ? (params.epochs / 30) : 1;
        if (epoch % print_interval == 0 || epoch == params.epochs - 1) {
            //Calculate test only 10 times - for now
            double epoch_test_loss = 0.0;
            if (test_inputs.get_rows_nb() > 0) {
                Matrix current_test_data = test_inputs;
                for (const auto& layer : layers) {
                    current_test_data = layer->feedforward(current_test_data);
                }
                epoch_test_loss = loss_function->calculate(current_test_data, test_targets);
            }

            std::cout << "Epoch [" << epoch + 1 << "/" << params.epochs << "] - Loss: " << epoch_loss << " | Test Loss: " << epoch_test_loss << std::endl;
            //Callback

            EpochStats stats(epoch + 1, epoch_loss, epoch_test_loss);

            bool keep_going = on_epoch_end(stats);
            if (!keep_going) {
                std::cout << "Training interrupted by user!" << std::endl;
                break;
            }

        }

    }
}

void NeuralNetwork::save(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    size_t num_layers = layers.size();
    out.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));

    for (const auto& layer : layers) {
        std::string name = layer->get_layer_name();
        size_t name_length = name.size();
        out.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
        out.write(name.c_str(), name_length);

        size_t in_nb = layer->get_input_nb();
        size_t out_nb = layer->get_output_nb();
        out.write(reinterpret_cast<const char*>(&in_nb), sizeof(in_nb));
        out.write(reinterpret_cast<const char*>(&out_nb), sizeof(out_nb));

        layer->save(out);
    }
}

void NeuralNetwork::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    layers.clear();

    size_t layer_nb;
    in.read(reinterpret_cast<char*>(&layer_nb), sizeof(layer_nb));

    for (size_t i = 0; i < layer_nb; ++i) {
        size_t name_length;
        in.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
        std::string name(name_length, '\0');
        in.read(&name[0], name_length);

        size_t in_nb, out_nb;
        in.read(reinterpret_cast<char*>(&in_nb), sizeof(in_nb));
        in.read(reinterpret_cast<char*>(&out_nb), sizeof(out_nb));

        LayerUI config;
        config.inputs = (int)in_nb;
        config.outputs = (int)out_nb;

        std::unique_ptr<Layer> layer = LayerMaker::create_by_name(name, i, config);

        if (!layer) {
            throw std::runtime_error("NeuralNetwork::load - Unknown layer: " + name);
        }

        layer->load(in);
        add_layer(std::move(layer));
    }
}