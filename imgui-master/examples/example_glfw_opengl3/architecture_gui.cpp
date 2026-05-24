#include "architecture_gui.h"
#include "layer_maker.h"
#include "app_gui.h"
#include "imgui.h"
#include <functional>

void show_architecture_settings(AppState& state) {
    // Show reset button and disable architecture configuration when loaded model
    if (state.is_architecture_locked) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Model loaded. Architecture is locked.");
        if (ImGui::Button("RESET ARCHITECTURE", ImVec2(-FLT_MIN, 30))) {
            state.is_architecture_locked = false;
            //state.nn.clear_layers();
            state.loss_history.clear();
            state.training_logs.clear();
        }
        ImGui::Spacing();
    }

    if (state.is_architecture_locked) ImGui::BeginDisabled();

    //EMBEDDING info
    size_t projected_in_dim = 0;

    if (state.dataset.has_value()) {
        size_t raw_inputs = state.dataset->input_data.get_columns_nb();
        size_t categorical_cols = state.embed_configs.size();
        projected_in_dim = raw_inputs;

        for (const auto& conf : state.embed_configs) {
            projected_in_dim += (conf.embed_dim - 1);
        }

        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Input Stage (Auto-Configured)");
        ImGui::BeginChild("InputStage", ImVec2(0, 60), true);
        ImGui::Text("Raw Features: %zu (%zu categorical)", raw_inputs, categorical_cols);

        if (categorical_cols > 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "-> TabularEmbeddingLayer injected. Expanded Input: %zu", projected_in_dim);
        }
        else {
            ImGui::TextDisabled("-> Pure numerical data. No embeddings needed.");
        }
        ImGui::EndChild();
        ImGui::Spacing();
    }

    // Dynamic table for layers configuration
    if (ImGui::BeginTable("LayersTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("In", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Out", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Params");
        ImGui::TableHeadersRow();

        // Look for the last configurable layer to set the out size of the table properly
        int last_configurable_idx = -1;
        for (int j = 0; j < (int)state.gui_layers.size(); ++j) {
            std::string l_name = state.layer_names[state.gui_layers[j].type_index];
            if (LayerMaker::registry.at(l_name) == LayerType::Dense) { // When implemented more layer types add here
                last_configurable_idx = j;
            }
        }

        //Set inputs and outputs for each layer based on the dataset and the previous layer configuration.
        // This is done before drawing the table to ensure that the values are correct when displayed. 
        for (size_t i = 0; i < state.gui_layers.size(); ++i) {
            std::string current_layer_name = state.layer_names[state.gui_layers[i].type_index];
            LayerType current_type = LayerMaker::registry.at(current_layer_name);

            if (i == 0) {
                if (state.dataset.has_value() && state.dataset->input_data.get_columns_nb() > 0) {
                    state.gui_layers[i].inputs = (int)projected_in_dim;
                }
            }
            else {
                state.gui_layers[i].inputs = state.gui_layers[i - 1].outputs;
            }
            if (current_type != LayerType::Dense) {
                state.gui_layers[i].outputs = state.gui_layers[i].inputs;
            }
            else if (static_cast<int>(i) == last_configurable_idx && state.dataset.has_value() && state.dataset->output_data.get_columns_nb() > 0) {
                state.gui_layers[i].outputs = (int)state.dataset->output_data.get_columns_nb();
            }
        }

        //Draw table rows
        for (size_t i = 0; i < state.gui_layers.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();

            std::string current_layer_name = state.layer_names[state.gui_layers[i].type_index];
            LayerType current_type = LayerMaker::registry.at(current_layer_name);

            // 1. TYP WARSTWY
            ImGui::TableSetColumnIndex(0);
            ImGui::SetNextItemWidth(-FLT_MIN);
            ImGui::Combo("##type", &state.gui_layers[i].type_index, state.layer_names.data(), (int)state.layer_names.size());

            // 2. WEJŒCIA (In)
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (current_type == LayerType::Dense && i == 0 && !state.dataset.has_value()) {
                ImGui::InputInt("##in", &state.gui_layers[i].inputs, 0);
            }
            else if (i == 0 && state.dataset.has_value()) {
                ImGui::TextDisabled("%d", state.gui_layers[i].inputs);
            }
            else {
                ImGui::TextDisabled("%d", state.gui_layers[i].inputs);
            }

            // 3. WYJŒCIA (Out)
            ImGui::TableSetColumnIndex(2);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (current_type  == LayerType::Dense) {
                if (i == state.gui_layers.size() - 1 && state.dataset.has_value()) {
                    ImGui::TextDisabled("%d (Cel)", state.gui_layers[i].outputs);
                }
                else {
                    ImGui::InputInt("##out", &state.gui_layers[i].outputs, 0);
                }
            }
            else {
                ImGui::TextDisabled("%d", state.gui_layers[i].outputs);
            }

            // 4. PARAMETRY
            ImGui::TableSetColumnIndex(3);
            ImGui::SetNextItemWidth(-FLT_MIN);
            // Indeks 3 w naszym systemie to "Dropout"
            if (state.gui_layers[i].type_index == 3) {
                ImGui::SliderFloat("Rate", &state.gui_layers[i].dropout_rate, 0.0f, 1.0f, "%.2f");
            }
            else {
                ImGui::TextDisabled("brak");
            }

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    // Przyciski zarz¹dzania tabel¹
    if (ImGui::Button("+ add layer", ImVec2(100, 30))) {
        state.gui_layers.push_back(LayerUI()); 
    }
    ImGui::SameLine();
    if (!state.gui_layers.empty()) {
        if (ImGui::Button("- remove last", ImVec2(100, 30))) {
            state.gui_layers.pop_back();
        }
    }

    if (state.is_architecture_locked) ImGui::EndDisabled();

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    // Hiperparametry
    ImGui::PushItemWidth(80);
    ImGui::InputFloat("Learning Rate", &state.hyperparams.learning_rate, 0.001f, 0.01f, "%.4f");
    ImGui::InputInt("Epochs",  &state.hyperparams.epochs);
    ImGui::InputInt("Batch Size", &state.hyperparams.batch_size);
    ImGui::Spacing();
    ImGui::Separator();

    ImGui::Text("Loss Function Settings");
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::Combo("##loss_fn", &state.selected_loss_idx, state.loss_names.data(), (int)state.loss_names.size());
    ImGui::PopItemWidth();
    ImGui::Spacing();
    ImGui::Spacing();
        
    // --- PRZYCISK TRAIN ---
    if (state.is_training) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.2f, 0.2f, 1.0f));
        if (ImGui::Button("STOP TRAINING", ImVec2(-FLT_MIN, 50))) {
            state.stop_training = true;
        }
        ImGui::PopStyleColor();
    }
    

    else {
        bool can_train = state.dataset.has_value() && state.gui_layers.size() >0;

        if (!can_train) ImGui::BeginDisabled();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
        auto train_button = ImGui::Button("TRAIN", ImVec2(-FLT_MIN, 50));
        ImGui::PopStyleColor();

        if (!can_train) {
            ImGui::EndDisabled();
            ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Load, fit and transform data to train.");
        }
        if (train_button && can_train) {
            if (!state.dataset) {
                ImGui::Text("No transformed dataset loaded!");
                return;
            }
            //Create new architecture if no loaded model
            if (!state.is_architecture_locked)
            {
                state.nn.clear_layers();

                size_t layer_id_counter = 0;
                size_t current_in_dim = state.dataset->input_data.get_columns_nb();

                // --- AUTOMATICALLY ADD EMBEDDING ---
                if (!state.embed_configs.empty()) {
                    auto embed_layer = std::make_unique<TabularEmbeddingLayer>(
                        layer_id_counter++, current_in_dim, state.embed_configs
                    );

                    current_in_dim = embed_layer->get_output_nb();
                    embed_layer->initialize(); // Initialize embedding layer before adding to the network
                    state.nn.add_layer(std::move(embed_layer));
                }
                //--- ADD REST OF THE LAYERS ---
                for (size_t i = 0; i < state.gui_layers.size(); ++i) {
                    std::string name = state.layer_names[state.gui_layers[i].type_index];
                    auto layer = LayerMaker::create_by_name(name, i, state.gui_layers[i]);
                    if (layer) {
                        layer->initialize();
                        state.nn.add_layer(std::move(layer));
                    }
                }
                //Clear logs from previous trainings only when creating new architecture
                state.loss_history.clear();
                state.training_logs.clear();
                state.test_loss_history.clear();
            }

            state.is_architecture_locked = true;

            state.nn.set_loss(LossFuncMaker::create_by_name(state.loss_names[state.selected_loss_idx]));
            state.is_training = true;
            state.stop_training = false;

            state.loss_history.reserve(state.loss_history.size()+ state.hyperparams.epochs);
            state.training_logs.reserve(state.training_logs.size() + state.hyperparams.epochs);
            state.test_loss_history.reserve(state.hyperparams.epochs); //Maybe divide by 10

            std::thread([&state]() {
                std::function<bool(EpochStats)> on_epoch_end = [&state](EpochStats stats) {
                    std::lock_guard<std::mutex> lock(state.gui_mutex);
                    state.loss_history.push_back(static_cast<float>(stats.loss));
                    state.test_loss_history.push_back(static_cast<float>(stats.test_loss));
                    state.training_logs.push_back("Epoch [" + std::to_string(stats.epoch) + "] - Loss: " + std::to_string(stats.loss)+ "| Test loss: "+std::to_string(stats.test_loss));

                    //Stop by user
                    if (state.stop_training) {
                        state.training_logs.push_back("--- STOPPED BY USER ---");
                        return false; 
                    }
                    return true;
                 };
                state.nn.train(state.dataset.value(), state.hyperparams, on_epoch_end);
                state.is_training = false;
                }).detach();
        }
    }

    //--- LOAD/SAVE MODEL ---
    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
    ImGui::Text("Model save & load (.bin)");

    ImGui::InputText("File Path##net", state.network_filepath, sizeof(state.network_filepath));

    // save button
    if (ImGui::Button("Save Model", ImVec2(120, 0))) {
        try {
            state.nn.save(state.network_filepath);
            state.network_status_msg = "Successfully saved to: " + std::string(state.network_filepath);
        }
        catch (const std::exception& e) {
            state.network_status_msg = std::string("Save error: ") + e.what();
        }
    }

    ImGui::SameLine();

    // load button
    if (ImGui::Button("Load Model", ImVec2(120, 0))) {
        try {
            state.nn.load(state.network_filepath);
            state.gui_layers.clear();
            const auto& loaded_layers = state.nn.get_layers();

            for (const auto& layer : loaded_layers) {
                LayerUI config;
                std::string name = layer->get_layer_name();

                if (name == "TabularEmbedding") continue; //skip embedding

                for (int j = 0; j < (int)state.layer_names.size(); ++j) {
                    if (std::string(state.layer_names[j]) == name) {
                        config.type_index = j;
                        break;
                    }
                }

                config.inputs = (int)layer->get_input_nb();
                config.outputs = (int)layer->get_output_nb();

                state.gui_layers.push_back(config);
            }
            state.is_architecture_locked = true;
            state.network_status_msg = "Successfully loaded from: " + std::string(state.network_filepath);
        }
        catch (const std::exception& e) {
            state.network_status_msg = std::string("Load error: ") + e.what();
        }
    }

    if (!state.network_status_msg.empty()) {
        if (state.network_status_msg.find("error") != std::string::npos) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.network_status_msg.c_str());
        }
        else {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", state.network_status_msg.c_str());
        }
    }
}
