#include "predict_GUI.h"
#include "app_gui.h"
#include "imgui.h"
#include <string>

void show_prediction_form(AppState& state) {
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Enter data for prediction:");
    ImGui::Spacing();

    //predict_num_inputs is map where we store gui inputs for numerical columns - the key is the column index
    //predict_cat_inputs is map where we store gui inputs for categorical columns - the key is the column index

    //The idea is to store the user input in two maps where the key is the column index, transform accordingly and create one-row input Matrix for prediction

    const auto& cols = state.prediction_preprocessor.get_columns();
    for (const auto& col : cols) {
        if (!col->include_column || col->is_target_column) continue;

        ImGui::PushID(static_cast<int>(col->get_index()));

        if (auto* num_col = dynamic_cast<NumericalColumn*>(col.get())) {
            ImGui::InputFloat(col->name.c_str(), &state.predict_num_inputs[col->get_index()]);//Input changes the value in the map
        }
        else if (auto* cat_col = dynamic_cast<CategoricalColumn*>(col.get())) {
            const auto& categories = cat_col->get_categories();
            int& current_idx = state.predict_cat_inputs[col->get_index()];

            if (current_idx < 0 || current_idx >= categories.size()) current_idx = 0;

            const char* cat_preview = categories.empty() ? "" : categories[current_idx].c_str();
            if (ImGui::BeginCombo(col->name.c_str(), cat_preview)) {
                for (int i = 0; i < categories.size(); ++i) {
                    bool is_sel = (current_idx == i);
                    if (ImGui::Selectable(categories[i].c_str(), is_sel)) {
                        current_idx = i;
                    }
                    if (is_sel) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }
        ImGui::PopID();
    }

    ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
    if (ImGui::Button("PREDICT", ImVec2(-FLT_MIN, 50))) {
        try {
            size_t input_size = 0;
            for (const auto& c : cols) {
                if (c->include_column && !c->is_target_column) input_size++;
            }

            //Build 1 x n matrix
            Matrix inference_input(1, input_size);
            size_t current_in_idx = 0;

            //Transform all given inputs
            for (const auto& c : cols) {
                if (!c->include_column || c->is_target_column) continue;
                //Categorical
                if (auto* cat_col = dynamic_cast<CategoricalColumn*>(c.get())) {
                    int cat_idx = state.predict_cat_inputs[c->get_index()];
                    std::string cat_str = cat_col->get_categories()[cat_idx];
                    inference_input(0, current_in_idx) = c->transform(cat_str);
                }
                //Numerical
                else {
                    float val = state.predict_num_inputs[c->get_index()];
                    inference_input(0, current_in_idx) = c->transform(std::to_string(val));
                }
                current_in_idx++;
            }

            Matrix output = state.prediction_nn.predict(inference_input);

            // Denormalise the prediction
            auto denormalized_results = state.prediction_preprocessor.inverse_transform_prediction(output);

            state.predict_results.clear();
            auto target_names = state.prediction_preprocessor.get_target_cols_names();

            for (size_t out_idx = 0; out_idx < output.get_columns_nb(); ++out_idx) {
                std::string t_name = (out_idx < target_names.size()) ? target_names[out_idx] : "Output " + std::to_string(out_idx);

                state.predict_results.push_back(t_name + ": " + denormalized_results[out_idx]);
            }
        }
        catch (const std::exception& e) {
            state.predict_results.clear();
            state.predict_results.push_back(std::string("ERROR: ") + e.what());
        }
    }
    ImGui::PopStyleColor();
}

void show_prediction_results(AppState& state) {
    ImGui::BeginChild("Stats", ImVec2(0, 0), true);
    if (state.predict_results.empty()) {
        ImGui::TextDisabled("Run inference to see results...");
    }
    else {
        for (const auto& res : state.predict_results) {
            if (res.find("ERROR") != std::string::npos) {
                ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "%s", res.c_str());
            }
            else {
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "> %s", res.c_str());
            }
            ImGui::Spacing();
        }
    }
    ImGui::EndChild();
}
