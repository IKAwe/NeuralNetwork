#include "app_gui.h"
#include "imgui.h"
#include "predict_GUI.h"
#include "csv_loader.h"
#include "layer_maker.h"
#include <filesystem>
#include "preprocessor_gui.h"
#include "architecture_gui.h"

// -----------------------------------------------------------------
// ZAK£ADKA TRAIN
// -----------------------------------------------------------------
void AppGUI::renderTrainTab() {
    if (ImGui::BeginTable("TrainLayout", 3, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        // --- Kolumna 1: Dane ---
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Preprocessing");
        ImGui::Separator();

        //Add refresh button to not look for files every iteration of render
        // ---Dropdown do wyboru pliku CSV-------
        if (state.csv_files.empty() && state.selected_file_idx == -1) {
            state.csv_files = find_files_by_extension(".csv");
            if (!state.csv_files.empty()) {
                state.selected_file_idx = 0; //Default is first file
            }
        }
        if (ImGui::Button("Refresh")) {
            state.csv_files = find_files_by_extension(".csv");
            state.selected_file_idx = state.csv_files.empty() ? -1 : 0;
        }

        ImGui::SameLine();

        const char* preview_value = (state.selected_file_idx >= 0 && state.selected_file_idx < state.csv_files.size())
            ? state.csv_files[state.selected_file_idx].c_str() : "No CSV files found...";

        if (ImGui::BeginCombo("##csv_combo", preview_value)) {
            for (int n = 0; n < state.csv_files.size(); n++) {
                const bool is_selected = (state.selected_file_idx == n);
                if (ImGui::Selectable(state.csv_files[n].c_str(), is_selected)) {
                    state.selected_file_idx = n;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();
        if (state.is_loading_csv) {
            ImGui::BeginDisabled();
            ImGui::Button("LOADING CSV... PLEASE WAIT", ImVec2(-FLT_MIN, 30));
            ImGui::EndDisabled();
        }
        else if (ImGui::Button("Load & initialize preprocessor", ImVec2(-FLT_MIN, 30))) {
            if(!state.csv_files.empty() && state.selected_file_idx >= 0) {

                state.is_loading_csv = true; // Block button
                std::string filepath = state.csv_files[state.selected_file_idx];

                std::thread([this, filepath]() {
                    try {
                        auto loaded_data = load_csv(filepath);

                        std::lock_guard<std::mutex> lock(state.gui_mutex);
                        state.raw_data = std::move(loaded_data);
                        state.preprocessor.initialize_from_data(state.raw_data);
                        state.is_fitted = false;

                        state.is_loading_csv = false;
                        //Maybe add status message
                    }
                    catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lock(state.gui_mutex);
                        state.is_loading_csv = false;
                    }
                    }).detach();
            }
        }
        show_preprocessor_settings(state);
        // --- Kolumna 2: Architektura ---
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Architecture");
        ImGui::Separator();
        show_architecture_settings(state);

        // --- Kolumna 3: Wyniki ---
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("Training Process");
        ImGui::Separator();
        {
            std::lock_guard<std::mutex> lock(state.gui_mutex);

            ImGui::BeginChild("Logs", ImVec2(0, 150), true);
            for (const auto& log : state.training_logs) {
                ImGui::TextUnformatted(log.c_str());
            }
            if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
                ImGui::SetScrollHereY(1.0f);
            }
            ImGui::EndChild();

            if (!state.loss_history.empty()) {
                ImGui::PlotLines("Loss", state.loss_history.data(), (int)state.loss_history.size(),
                    0, nullptr, FLT_MAX, FLT_MAX, ImVec2(-FLT_MIN, 150));
            }
            else {
                ImGui::TextDisabled("Waiting for training to start...");
            }
        }

        ImGui::EndTable();
    }
}

// -----------------------------------------------------------------
// ZAK£ADKA PREDICT
// -----------------------------------------------------------------
void AppGUI::renderPredictTab() {
    if (ImGui::BeginTable("PredictLayout", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        // --- Kolumna 1: Akcje ---
        ImGui::TableSetColumnIndex(0);

        if (ImGui::Button("Refresh Files in Folder", ImVec2(-FLT_MIN, 30))) {
            state.bin_files = find_files_by_extension(".bin");
            state.json_files = find_files_by_extension(".json");
            state.selected_model_idx = state.bin_files.empty() ? -1 : 0;
            state.selected_json_idx = state.json_files.empty() ? -1 : 0;
            state.predict_status_msg = "Files list refreshed.";
        }
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        // 1. DROPDOWN: Wybór Modelu (.bin)
        const char* model_preview = (state.selected_model_idx >= 0 && state.selected_model_idx < state.bin_files.size())
            ? state.bin_files[state.selected_model_idx].c_str() : "Select .bin file...";

        if (ImGui::BeginCombo("Neural Network Model", model_preview)) {
            for (int n = 0; n < state.bin_files.size(); n++) {
                const bool is_selected = (state.selected_model_idx == n);
                if (ImGui::Selectable(state.bin_files[n].c_str(), is_selected)) {
                    state.selected_model_idx = n;
                    std::string filepath = state.bin_files[n];

                    // Make thread to load model
                    std::thread([this, filepath]() {
                        try {
                            state.nn.load(filepath);
                            std::lock_guard<std::mutex> lock(state.gui_mutex);
                            state.predict_status_msg = "Model loaded: " + filepath;
                        }
                        catch (const std::exception& e) {
                            std::lock_guard<std::mutex> lock(state.gui_mutex);
                            state.predict_status_msg = "Model Load Error: " + std::string(e.what());
                        }
                    }).detach();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        // 2. DROPDOWN: Wybór Preprocesora (.json)
        const char* json_preview = (state.selected_json_idx >= 0 && state.selected_json_idx < state.json_files.size())
            ? state.json_files[state.selected_json_idx].c_str() : "Select .json config...";

        if (ImGui::BeginCombo("Preprocessor Config", json_preview)) {
            for (int n = 0; n < state.json_files.size(); n++) {
                const bool is_selected = (state.selected_json_idx == n);
                if (ImGui::Selectable(state.json_files[n].c_str(), is_selected)) {
                    state.selected_json_idx = n;
                    std::string filepath = state.json_files[n];

                    std::thread([this, filepath]() {
                        try {
                            state.preprocessor.load(filepath);
                            state.is_fitted = true;
                            std::lock_guard<std::mutex> lock(state.gui_mutex);
                            state.predict_status_msg = "Preprocessor config loaded: " + filepath;
                        }
                        catch (const std::exception& e) {
                            std::lock_guard<std::mutex> lock(state.gui_mutex);
                            state.predict_status_msg = "Config Load Error: " + std::string(e.what());
                        }
                        }).detach();
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        // --- PREDICT ---
        
        show_prediction_form(state);


        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Network Outputs");
        ImGui::Separator();

        show_prediction_results(state);

        ImGui::EndTable();
    }
}

// -----------------------------------------------------------------
// G£ÓWNA RAMA OKNA
// -----------------------------------------------------------------
void AppGUI::render() {
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1000, 600), ImGuiCond_FirstUseEver);

    ImGui::Begin("Neural Network Builder", nullptr);

    if (ImGui::BeginTabBar("MainTabs")) {
        if (ImGui::BeginTabItem("Train")) {
            renderTrainTab();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Predict")) {
            renderPredictTab();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}
