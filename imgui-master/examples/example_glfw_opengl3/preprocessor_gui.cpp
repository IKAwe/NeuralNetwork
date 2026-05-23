#include "preprocessor_gui.h"
#include "data_preprocessor.h"
#include "imgui.h"
#include "app_gui.h"


void show_col_statistics(const std::unique_ptr<Column>& col) {
    ImGui::TableSetColumnIndex(3);

    if (!col->include_column ) {
        ImGui::TextDisabled("Excluded");
        ImGui::TableSetColumnIndex(4);
        ImGui::TextDisabled("-");
        return;
    }

    auto* num_col = dynamic_cast<NumericalColumn*>(col.get());
    auto* cat_col = dynamic_cast<CategoricalColumn*>(col.get());

    if (num_col) {
        ImGui::Text("%.2f", num_col->get_mean());
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%.2f", num_col->get_std_dev());
    }
    else if (cat_col) {
        const auto& cats = cat_col->get_categories();
        std::string preview = "";

        int show_count = std::min(2, (int)cats.size());
        for (int j = 0; j < show_count; ++j) {
            preview += cats[j];
            preview += (j < show_count - 1) ? ", " : "";
        }
        if (cats.size() > 1) {
            preview += " (" + std::to_string(cats.size()) + ")";
        }

        ImGui::TextUnformatted(preview.c_str());

        ImGui::TableSetColumnIndex(4);
        ImGui::TextDisabled("-");
    }
}

void show_initialized_col(std::unique_ptr<Column>& col) {
    //  Include checkbox
    ImGui::TableSetColumnIndex(3);
    ImGui::PushID(col->get_index());
    ImGui::Checkbox("##inc", &col->include_column);

    // Is Target checkbox
    ImGui::TableSetColumnIndex(4);
    if (ImGui::Checkbox("##target", &col->is_target_column)) {
    }
    ImGui::PopID();
}



void show_preprocessor_settings(AppState& state) {
    auto& columns = state.preprocessor.get_columns_mutable();

    if (columns.empty()) {
        ImGui::TextDisabled("No data loaded or the file was empty. Please select and load right CSV file.");
        return;
    }

    ImGui::SeparatorText("Column Settings");

    static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable;

    if (ImGui::BeginTable("ColumnsTable", 5, flags)) {
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 30.0f);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Type");
        if (state.preprocessor.is_fitted()) {
            ImGui::TableSetupColumn("Mean / Categories");
            ImGui::TableSetupColumn("Std Dev");
        }
        else {
            ImGui::TableSetupColumn("Include", ImGuiTableColumnFlags_WidthFixed, 60.0f);
            ImGui::TableSetupColumn("Is Target", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        }
        ImGui::TableHeadersRow();

        for (int i = 0; i < columns.size(); i++) {
            auto& col = columns[i]; // col to std::unique_ptr<Column>&

            ImGui::TableNextRow();

            // 0. ID
            ImGui::TableSetColumnIndex(0);
            if(col->is_target_column)
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%d", (int)col->get_index());
            else
                ImGui::Text("%d", (int)col->get_index());

            // 1. Name
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(col->name.c_str());

            // 2. Type
            ImGui::TableSetColumnIndex(2);
            if (dynamic_cast<NumericalColumn*>(col.get())) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Numerical");
            }
            else {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Categorical");
            }

            if (state.preprocessor.is_fitted()) {
                show_col_statistics(col);
            }
            else {
                show_initialized_col(col);
            }
        }
        ImGui::EndTable();
    }

    ImGui::Spacing();

    // ========FIT=TRANSFORM BUTTONS========
    if (!state.preprocessor.is_fitted()) {
        if (state.is_fitting) {
            ImGui::BeginDisabled();
            ImGui::Button("FITTING THE DATA... PLEASE WAIT", ImVec2(-FLT_MIN, 40));
            ImGui::EndDisabled();
        }
        else {
            if (ImGui::Button("FIT DATA", ImVec2(-FLT_MIN, 40))) {
                state.is_fitting = true;

                std::thread([&state]() {
                    state.preprocessor.fit(state.raw_data);
                    state.is_fitting = false;
                    }).detach();
            }
        }
    }
    else {
        if (state.is_transforming) {
            ImGui::BeginDisabled();
            ImGui::Button("TRANSFORMING... PLEASE WAIT", ImVec2(-FLT_MIN, 40));
            ImGui::EndDisabled();
        }
        else {
            if (ImGui::Button("TRANSFORM DATA", ImVec2(-FLT_MIN, 40))) {
                state.is_transforming = true;

                std::thread([&state]() {
                    auto result = state.preprocessor.transform(state.raw_data);
                    state.dataset = std::move(result);
                    //==Make embedding configs for each categorical column=====
                    state.embed_configs.clear();
                    const auto& in_cat_counts = state.dataset->metadata.input_category_counts;
                    for (size_t i = 0; i < in_cat_counts.size(); ++i) {
                        int vocab = in_cat_counts[i];
                        if (vocab > 0) {
                            size_t embed_dim = std::max((size_t)1, std::min((size_t)50, (size_t)(vocab)/2)); //EMBEDDING DIMENSION
                            state.embed_configs.push_back({ i, (size_t)vocab, embed_dim });
                        }
                    }
                    //=========================================================
                    state.is_transforming = false;
                    }).detach();
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Text("Preprocessor Configuration (JSON)");

    // Text input for file path for saving/loading preprocessor.
    ImGui::InputText("File Path##prep", state.preprocessor_filepath, sizeof(state.preprocessor_filepath));

    // Przycisk ZAPISZ
    if (state.preprocessor.is_fitted())
    {
        if (ImGui::Button("Save Config")) {
            try {
                state.preprocessor.save(state.preprocessor_filepath);
                state.preprocessor_status_msg = "Successfully saved to: " + std::string(state.preprocessor_filepath);
            }
            catch (const std::exception& e) {
                state.preprocessor_status_msg = std::string("Save error: ") + e.what();
            }
        }
    }

    ImGui::SameLine();

    // Przycisk WCZYTAJ
    if (ImGui::Button("Load Config")) {
        try {
            state.preprocessor.load(state.preprocessor_filepath);
            state.preprocessor_status_msg = "Successfully loaded from: " + std::string(state.preprocessor_filepath);
        }
        catch (const std::exception& e) {
            state.preprocessor_status_msg = std::string("Load error: ") + e.what();
        }
    }

    //Show status
    if (!state.preprocessor_status_msg.empty()) {
        if (state.preprocessor_status_msg.find("error") != std::string::npos) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", state.preprocessor_status_msg.c_str());
        }
        else {
            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", state.preprocessor_status_msg.c_str());
        }
    }
}

