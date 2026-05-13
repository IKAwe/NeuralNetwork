#include "architecture_gui.h"
#include "layer_maker.h"
#include "app_gui.h"
#include "imgui.h"
#include <functional>

void show_architecture_settings(AppState& state) {

    // Rysowanie dynamicznej tabeli warstw (zastêpuje stary kod)
    if (ImGui::BeginTable("LayersTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Type");
        ImGui::TableSetupColumn("In", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Out", ImGuiTableColumnFlags_WidthFixed, 40.0f);
        ImGui::TableSetupColumn("Params");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < state.gui_layers.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();

            // Automatyczne spinanie wejœæ z wyjœciem poprzedniej warstwy
            if (i == 0) {
                state.gui_layers[i].inputs = state.dataset ? (int)state.dataset->input_data.get_columns_nb() : 0;
            }
            else if (i > 0) {
                state.gui_layers[i].inputs = state.gui_layers[i - 1].outputs;
                if (i == state.gui_layers.size() - 1) {
                    state.gui_layers[i].outputs = state.dataset ? (int)state.dataset->output_data.get_columns_nb() : 0;
                }
            }

            // 1. TYP WARSTWY
            ImGui::TableSetColumnIndex(0);
            ImGui::SetNextItemWidth(-FLT_MIN);
            // Pobieramy nazwy z naszej Fabryki
            ImGui::Combo("##type", &state.gui_layers[i].type_index, state.layer_names.data(), (int)state.layer_names.size());

            // 2. WEJŒCIA (In)
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-FLT_MIN);

            std::string current_layer_name = state.layer_names[state.gui_layers[i].type_index];
            LayerType current_type = LayerMaker::registry.at(current_layer_name);

            if (current_type == LayerType::Dense) {
                if ((i == 0) && state.dataset.has_value()) {
                    ImGui::TextDisabled("%d (Dane)", state.gui_layers[i].inputs);
                }
                else if (i == 0) {
                    ImGui::InputInt("##in", &state.gui_layers[i].inputs, 0);
                }
                else {
                    ImGui::TextDisabled("%d (Auto)", state.gui_layers[i].inputs); 
                }
            }
            else {
                ImGui::Text(" - ");
                state.gui_layers[i].outputs = state.gui_layers[i].inputs; // Aktywacje przepuszczaj¹ rozmiar
            }

            // 3. WYJŒCIA (Out)
            ImGui::TableSetColumnIndex(2);
            ImGui::SetNextItemWidth(-FLT_MIN);
            if (current_type  == LayerType::Dense) {
                // ZMIANA: Zablokuj edycjê dla ostatniej warstwy, jeœli wymusza to dataset
                if (i == state.gui_layers.size() - 1 && state.dataset.has_value()) {
                    ImGui::TextDisabled("%d (Cel)", state.gui_layers[i].outputs);
                }
                else {
                    ImGui::InputInt("##out", &state.gui_layers[i].outputs, 0);
                }
            }
            else {
                ImGui::Text(" - ");
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
        state.gui_layers.push_back(LayerUI()); // Dodaje domyœln¹ strukturê konfiguracyjn¹
    }
    ImGui::SameLine();
    if (!state.gui_layers.empty()) {
        if (ImGui::Button("- remove last", ImVec2(100, 30))) {
            state.gui_layers.pop_back();
        }
    }

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
        ImGui::BeginDisabled();
        ImGui::Button("TRAINING IN PROGRESS... PLEASE WAIT", ImVec2(-FLT_MIN, 40));
        ImGui::EndDisabled();
    }
    

    else {
        bool can_train = state.dataset.has_value();

        if (!can_train) ImGui::BeginDisabled();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
        auto train_button = ImGui::Button("TRAIN", ImVec2(-FLT_MIN, 50));
        ImGui::PopStyleColor();

        if (!can_train) {
            ImGui::EndDisabled();
            ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "Load, fit and transform data to train.s");
        }
        if (train_button && can_train) {
            if (!state.dataset) {
                ImGui::Text("No transformed dataset loaded!");
                return;
            }
            //Update layers
            state.nn.clear_layers();
            for (size_t i = 0; i < state.gui_layers.size(); ++i) {
                std::string name = state.layer_names[state.gui_layers[i].type_index];
                auto layer = LayerMaker::create_by_name(name, i, state.gui_layers[i]);

                if (layer) {
                    state.nn.add_layer(std::move(layer));
                }
            }
            state.nn.set_loss(LossFuncMaker::create_by_name(state.loss_names[state.selected_loss_idx]));
            state.is_training = true;

            std::thread([&state]() {
                std::function<void(EpochStats)> on_epoch_end = [&state](EpochStats stats) {
                    std::lock_guard<std::mutex> lock(state.gui_mutex);
                    state.loss_history.push_back(static_cast<float>(stats.loss));
                    state.training_logs.push_back("Epoch [" + std::to_string(stats.epoch) + "] - Loss: " + std::to_string(stats.loss));
                    };
                state.nn.train(state.dataset.value(), state.hyperparams, on_epoch_end);
                state.is_training = false;
                }).detach();
        }
    }
}
