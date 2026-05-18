#include "app_gui.h"
#include "imgui.h"
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
        if (state.csv_files.empty()) {
            state.csv_files = find_files_by_extension(".csv");
        }

        // Przygotowujemy etykietê aktualnie wybranego pliku
        const char* preview_value = state.csv_files.empty() ? "No files found" : state.csv_files[state.selected_file_idx].c_str();

        if (ImGui::BeginCombo("Select CSV", preview_value)) {
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

        if (ImGui::Button("Load & initialize preprocessor", ImVec2(-FLT_MIN, 30))) {
            if (!state.csv_files.empty()) {
                state.raw_data = load_csv(state.csv_files[state.selected_file_idx]);
                state.preprocessor.initialize_from_data(state.raw_data);
                state.is_fitted = false;
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
            state.csv_files = find_files_by_extension(".csv");
            state.selected_model_idx = state.bin_files.empty() ? -1 : 0;
            state.selected_json_idx = state.json_files.empty() ? -1 : 0;
            state.selected_csv_idx = state.csv_files.empty() ? -1 : 0;
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
                    std::thread([&state, filepath]() {
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

                    std::thread([&state, filepath]() {
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

        // 3. DROPDOWN: Wybór nowych danych (.csv)
        const char* csv_preview = (state.selected_csv_idx >= 0 && state.selected_csv_idx < state.csv_files.size())
            ? state.csv_files[state.selected_csv_idx].c_str() : "Select .csv dataset...";

        if (ImGui::BeginCombo("Input Data (CSV)", csv_preview)) {
            for (int n = 0; n < state.csv_files.size(); n++) {
                const bool is_selected = (state.selected_csv_idx == n);
                if (ImGui::Selectable(state.csv_files[n].c_str(), is_selected)) {
                    state.selected_csv_idx = n;
                }
                if (is_selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        // Zapis wyniku
        ImGui::InputTextWithHint("Output File", "e.g., predictions.csv", state.output_filename, IM_ARRAYSIZE(state.output_filename));
        ImGui::Spacing();

        // --- PRZYCISK PREDICT ---
        bool ready_to_predict = (state.selected_model_idx >= 0 && state.selected_json_idx >= 0 && state.selected_csv_idx >= 0);

        if (!ready_to_predict) ImGui::BeginDisabled();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));

        if (ImGui::Button("PREDICT & SAVE", ImVec2(-FLT_MIN, 50))) {
            std::string input_csv = state.csv_files[state.selected_csv_idx];
            std::string out_csv = state.output_filename;

            // TODO: Podpiêcie w³aœciwej logiki przewidywania i zapisu!
            // Wykonamy to w osobnym w¹tku w kolejnym kroku :)
            state.predict_status_msg = "[TODO] Uruchamianie predykcji na pliku: " + input_csv;
        }

        ImGui::PopStyleColor();
        if (!ready_to_predict) ImGui::EndDisabled();

        // Status
        if (!state.predict_status_msg.empty()) {
            if (state.predict_status_msg.find("Error") != std::string::npos) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.predict_status_msg.c_str());
            }
            else {
                ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.2f, 1.0f), "%s", state.predict_status_msg.c_str());
            }
        }

        // --- Kolumna 2: Statystyki ---
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Statistics & Logs");
        ImGui::Separator();
        ImGui::BeginChild("Stats", ImVec2(0, 0), true);
        ImGui::TextWrapped("Select the Neural Network binary, Preprocessor JSON config, and the CSV file containing new data to evaluate.");
        ImGui::TextWrapped("Once predictions are generated, statistics will appear here.");
        ImGui::EndChild();

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
