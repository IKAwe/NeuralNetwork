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

        // ---Dropdown do wyboru pliku CSV-------
        if (state.csv_files.empty()) {
            state.csv_files = find_csv_files();
        }

        // Etykieta aktualnie wybranego pliku
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
        ImGui::BeginChild("Logs", ImVec2(0, 150), true);
        ImGui::Text("Logs from training...");
        ImGui::EndChild();

        if (state.loss_history.empty()) { state.loss_history = { 0.9f, 0.5f, 0.2f, 0.1f }; }
        ImGui::PlotLines("Loss", state.loss_history.data(), state.loss_history.size(), 0, nullptr, 0.0f, 1.0f, ImVec2(-FLT_MIN, 150));

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
        if (ImGui::Button("Choose model file", ImVec2(-FLT_MIN, 40))) { /* TODO */ }
        ImGui::Spacing();
        if (ImGui::Button("Choose data file", ImVec2(-FLT_MIN, 40))) { /* TODO */ }
        ImGui::Spacing();
        ImGui::InputTextWithHint("##out", "output file name...", state.output_filename, IM_ARRAYSIZE(state.output_filename));
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
        if (ImGui::Button("PREDICT", ImVec2(-FLT_MIN, 50))) { /* TODO */ }
        ImGui::PopStyleColor();

        // --- Kolumna 2: Statystyki ---
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("Statistics");
        ImGui::Separator();
        ImGui::BeginChild("Stats", ImVec2(0, 0), true);
        ImGui::Text("Results of predictions...");
        ImGui::EndChild();

        ImGui::EndTable();
    }
}

// ---------------------------------------------------------------
// G£ÓWNA RAMA OKNA 
// ---------------------------------------------------------------
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
