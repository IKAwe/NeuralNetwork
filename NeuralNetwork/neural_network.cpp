#include "neural_network.h"
#include "loss_functions.h"
#include "layer_maker.h"
#include <iostream>
#include <chrono>
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
/**
 * @brief Get the output of the model for given input. The input should have the same number of columns as the input dimension of the first layer. The function will pass the input through each layer sequentially and return the final output. If the model has no layers, an exception will be thrown.
 * @param inputs The input data as a Matrix. Each row represents a sample, and each column represents a feature. The number of columns should match the input dimension of the first layer in the model.
 * @return The output of the model as a Matrix. Each row represents a sample, and each column represents an output feature. The number of columns will match the output dimension of the last layer in the model.
 */
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

/**
 * @brief Test the model on given input and targets. The function will first call predict to get the model's output for the input, and then calculate the loss using the specified loss function. The input and target dimensions should match the model's architecture. If no loss function is set, an exception will be thrown.
 * @param inputs The input data as a Matrix. Each row represents a sample, and each column represents a feature. The number of columns should match the input dimension of the first layer in the model.
 * @param targets The target data as a Matrix. Each row represents a sample, and each column represents a target feature. The number of columns should match the output dimension of the last layer in the model.
 * @return The calculated loss as a double.
 */
double NeuralNetwork::test(const Matrix& inputs, const Matrix& targets) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::test - No loss function was added!");
    }

    Matrix predictions = predict(inputs);

    double error = loss_function->calculate(predictions, targets);

    return error;
}

/**
 * @brief Train the model using the provided dataset and hyperparameters. The function will perform forward and backward passes through the model for a specified number of epochs, updating the model's parameters using the calculated gradients. The training will be done in batches, and the loss will be calculated for each batch. After each epoch, the average loss will be printed, and if a test set is provided, the test loss will also be calculated and printed. The function also accepts a callback that will be called at the end of each epoch with the epoch statistics, allowing for custom actions such as early stopping or logging. If no loss function is set, an exception will be thrown.
 * @param dataset The dataset to train the model on. It should contain input and output data, as well as optional test data.
 * @param params The hyperparameters for training, such as learning rate, batch size, and number of epochs.
 * @param on_epoch_end A callback function that will be called at the end of each epoch with the epoch statistics. The function should return true to continue training or false to stop early.
 */
void NeuralNetwork::train(const Dataset& dataset, const Hyperparams params, std::function<bool(EpochStats)> on_epoch_end) {
    if (!loss_function) {
        throw std::runtime_error("NeuralNetwork::train - Loss function was not added");
    }

    const Matrix& inputs = dataset.input_data;
    const Matrix& targets = dataset.output_data;
    size_t total_samples = inputs.get_rows_nb();

    const Matrix& test_inputs = dataset.test_inputs;
    const Matrix& test_targets = dataset.test_outputs;
    
    //Safety checks
    size_t train_input_rows = inputs.get_rows_nb();
    size_t test_input_rows = test_inputs.get_rows_nb();

    size_t train_output_rows = targets.get_rows_nb();
    size_t test_output_rows = test_targets.get_rows_nb();


    size_t train_input_cols = inputs.get_columns_nb();
    size_t test_input_cols = test_inputs.get_columns_nb();

    size_t train_output_cols = targets.get_columns_nb();
    size_t test_output_cols = test_targets.get_columns_nb();


    if (total_samples == 0) {
        throw std::runtime_error("NeuralNetwork::train - Dataset is empty");
    }
    if (train_output_cols == 0) {
        throw std::runtime_error("NeuralNetwork:train - No target column");
    }
    if (train_input_cols != layers.front()->get_input_nb()) {
        throw std::runtime_error(
            "NeuralNetwork::train - Input dimension mismatch. "
            "Dataset input columns: " + std::to_string(inputs.get_columns_nb()) +
            ", Expected: " + std::to_string(layers.front()->get_input_nb())
        );
	}
    if (train_output_cols != layers.back()->get_output_nb()) {
        throw std::runtime_error(
            "NeuralNetwork::train - Target dimension mismatch. "
            "Dataset target columns: " + std::to_string(train_output_cols) +
            ", Expected: " + std::to_string(layers.back()->get_output_nb())
        );
	}
    if (train_input_rows != train_output_rows) {
        throw std::runtime_error(
            "NeuralNetwork::train - Number of input samples does not match number of target samples. "
            "Input rows: " + std::to_string(train_input_rows) +
            ", Target rows: " + std::to_string(train_output_rows)
        );
    }
    // --- TEST SET VALIDATION (if present) ---
    bool test_present = (test_input_rows > 0 || test_output_rows > 0);
    if (test_present) {
        if (test_input_rows != test_output_rows) {
            throw std::runtime_error(
                "NeuralNetwork::train - Test set row count mismatch. "
                "test_inputs rows: " + std::to_string(test_input_rows) +
                ", test_targets rows: " + std::to_string(test_output_rows)
            );
        }

        if (test_input_rows > 0 &&
            test_input_cols != layers.front()->get_input_nb()) {

            throw std::runtime_error(
                "NeuralNetwork::train - Test input dimension mismatch. "
                "test_inputs columns: " + std::to_string(test_input_cols) +
                ", Expected: " + std::to_string(layers.front()->get_input_nb())
            );
        }
        if (test_output_rows > 0 &&
            test_output_cols != layers.back()->get_output_nb()) {

            throw std::runtime_error(
                "NeuralNetwork::train - Test target dimension mismatch. "
                "test_targets columns: " + std::to_string(test_output_cols) +
                ", Expected: " + std::to_string(layers.back()->get_output_nb())
            );
        }
    }
    if(params.batch_size <= 0){
        throw std::runtime_error("NeuralNetwork::train - Batch size must be greater than 0");
	}
    if (params.epochs <= 0) {
        throw std::runtime_error("NeuralNetwork::train - Number of epochs must be greater than 0");
    }
    //Preallocation for batching
    Matrix batch_inputs(params.batch_size, inputs.get_columns_nb());
    Matrix batch_targets(params.batch_size, targets.get_columns_nb());

	size_t print_nb = 20; // Print only 20 times during training, to avoid spamming the console
    size_t print_interval = (params.epochs >= print_nb) ? (params.epochs / print_nb) : 1;
    
    auto start_time = std::chrono::high_resolution_clock::now();

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

        
        if (epoch % print_interval == 0 || epoch == params.epochs - 1) {
            //Calculate test only 20 times - for now
            double epoch_test_loss = 0.0;
            if (test_present) {
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
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
	std::cout << "Training completed in " << duration << " seconds." << std::endl;
}
/**
 * @brief Save the model architecture and parameters to a binary file. The function will write the number of layers, followed by the name, input dimension, output dimension, and parameters of each layer in binary format. The file can later be loaded using the load function to restore the model. If the file cannot be opened for writing, an exception will be thrown.
 * @param filename The name of the file to save the model to.
 */
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
/**
 * @brief Load the model architecture and parameters from a binary file. The function will read the number of layers, followed by the name, input dimension, output dimension, and parameters of each layer in binary format. The existing model will be cleared before loading the new one. If the file cannot be opened for reading, or if the file format is invalid, an exception will be thrown.
 * @param filename The name of the file to load the model from.
 */
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