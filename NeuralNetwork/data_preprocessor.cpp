#include "data_preprocessor.h"
#include "data_structures.h"
#include <charconv>
#include <cmath>
#include <set>
#include <fstream>
#include <iostream>
#include <algorithm>

/**
 * @brief Fit the numerical column by calculating the mean and standard deviation of the values in the column. 
 This is done in two passes through the data - first to calculate the mean, and second to calculate the standard deviation. 
 The calculated mean and standard deviation are stored in the class members for later use in the transform() function.
 * @param data The data to fit the preprocessor on. The function assumes that the first row of the data contains column headers and starts processing from the second row.
 */
void NumericalColumn::fit(const StringMatrix& data) {
    double sum = 0.0;
    size_t count = 0;
    std::vector<double> values;
    values.reserve(data.get_rows_nb() - 1);

	//First pass: calculate mean
    for (size_t r = 1; r < data.get_rows_nb(); ++r) {
        std::string_view cell = data(r, index);
        double val = 0.0;
		//Use from_chars for efficient *char to double conversion
        auto [ptr, ec] = std::from_chars(cell.data(), cell.data() + cell.size(), val);

        if (ec == std::errc()) {
            values.push_back(val);
            sum += val;
            count++;
        }
    }

    if (count > 0) {
        mean = sum / count;

		// Second pass: calculate standard deviation
        double squares_sum = 0.0;
        for (double v : values) {
            squares_sum += std::pow(v - mean, 2);
        }
        std_dev = std::sqrt(squares_sum / count);

		// Avoid division by zero in transform() if all values are the same
        if (std_dev == 0.0) std_dev = 1.0;
    }
}
/**
 * @brief Transform a cell value by normalizing it using the mean and standard deviation calculated in the fit() function.
 * @param cell The cell to be transformed.
 * @return The normalized value of the cell.
 */
double NumericalColumn::transform(const std::string_view cell) {
    double val = 0.0;
    std::from_chars(cell.data(), cell.data() + cell.size(), val);
    return std::clamp((val - mean) / std_dev, -5.0, 5.0);
}

std::string NumericalColumn::inverse_transform(const double val) const {
    double original = (val * std_dev) + mean;
    return std::to_string(original);
}

json NumericalColumn::serialize() const {
    json j = Column::serialize();
	j["type"] = "Numerical";
    j["mean"] = mean;
    j["std_dev"] = std_dev;
    return j;
}
void NumericalColumn::deserialize(const json& j){
    Column::deserialize(j);
    mean = j.at("mean").get<double>();
    std_dev = j.at("std_dev").get<double>();
}

// --- CategoricalColumn Implementation ---


void CategoricalColumn::fit(const StringMatrix& data) {
	// Get all values as string_views to avoid unnecessary string constructions
    std::vector<std::string_view> raw_views;
    raw_views.reserve(data.get_rows_nb() - 1);

    for (size_t r = 1; r < data.get_rows_nb(); ++r) {
        if (!data(r, index).empty()) {
            raw_views.push_back(data(r, index));
        }
    }

	// Sort string_views to bring duplicates together
    std::sort(raw_views.begin(), raw_views.end());

	// Move unique values to the front of the vector and erase the rest
    auto last = std::unique(raw_views.begin(), raw_views.end());
    raw_views.erase(last, raw_views.end());

	// Create categories strings vector from unique string_views
    categories.clear();
    categories.reserve(raw_views.size());
    for (auto v : raw_views) {
        categories.emplace_back(v);
    }
}
/**
 * @brief Transform each cell to its category index (int). This is specifically made for embedding layer - if you don't add embedding layer as the first layer you will have poor results due to not normalised inputs.
 * @param cell 
 * @return 
 */
double CategoricalColumn::transform(const std::string_view cell) {
	//Since categories are sorted, we can use binary search to find the index of the category
    auto it = std::lower_bound(categories.begin(), categories.end(), cell);

	// Check if the category was found and return its index as a double. If not found, return -1.0.
    if (it != categories.end() && *it == cell) {
        if (scale_by_count) {
            return static_cast<double>(std::distance(categories.begin(), it)) / categories.size();
		}
        return static_cast<double>(std::distance(categories.begin(), it));
    }

    return -1.0;
}

std::string CategoricalColumn::inverse_transform(double val) const {
    if (scale_by_count) {
        val = val * categories.size();
	}
    int idx = static_cast<int>(std::round(val));
    if (idx >= 0 && idx < categories.size()) {
        return categories[idx];
    }
    return "UNKNOWN_CATEGORY";
}

json CategoricalColumn::serialize() const {
    json j = Column::serialize();
	j["type"] = "Categorical";
    j["categories"] = categories;
    j["scale_by_count"] = scale_by_count;
    return j;
}
void CategoricalColumn::deserialize(const json& j) {
    Column::deserialize(j);
    categories = j.at("categories").get<std::vector<std::string>>();
	scale_by_count = j.at("scale_by_count").get<bool>();
}


// --- DataPreprocessor Implementation ---
void DataPreprocessor::initialize_from_data(const StringMatrix& data) {
    columns.clear();
    fitted = false;
	if (data.get_rows_nb() < 2) return; // Not enough data to determine column types
    for (size_t c = 0; c < data.get_columns_nb(); ++c) {
        std::string name(data(0, c)); // Header

        // Check first row for types
        std::string_view sample = data(1, c);
        double cell_value;
        auto [ptr, ec] = std::from_chars(sample.data(), sample.data() + sample.size(), cell_value);

        if (ec == std::errc()) {
			// Number -> add NumericalColumn
            auto col = std::make_unique<NumericalColumn>(c, name);
            columns.push_back(std::move(col));
        }
        else {
			// Text -> add CategoricalColumn
            auto col = std::make_unique<CategoricalColumn>(c, name);
            columns.push_back(std::move(col));
        }
    }
}
void DataPreprocessor::fit(const StringMatrix& data) {
    fitted = false;
    for (auto& col : columns) {
		if (col->include_column) col->fit(data);
    }
    fitted = true;
}

Dataset DataPreprocessor::transform(const StringMatrix& data, double test_fraction) {
    if (!fitted) {
        throw std::logic_error("DataPreprocessor::transform called before fit() or load().");
    }
    size_t num_rows = data.get_rows_nb() - 1; //First row is the header
    std::vector<Column*> input_cols;
    std::vector<Column*> output_cols;
    std::vector<int> in_cat_counts;
    std::vector<int> out_cat_counts;

    size_t test_row_nb = static_cast<size_t>(num_rows * test_fraction);
    size_t train_row_nb = num_rows - test_row_nb;

    size_t total_output_width = 0;

	// Count how many input and output columns we have based on the is_target_column flag
    for (const auto& col : columns) {
		if (!col->include_column) continue; // Skip not included
        int cat_count = 0;
        if (auto cat_col = dynamic_cast<CategoricalColumn*>(col.get())) {
            cat_count = static_cast<int>(cat_col->get_categories().size());
        }
        if (col->is_target_column) {
            output_cols.push_back(col.get());
            out_cat_counts.push_back(cat_count);

            total_output_width += (cat_count > 0) ? cat_count : 1;
        }
        else {
            input_cols.push_back(col.get());
            in_cat_counts.push_back(cat_count);
        }
    }
    size_t total_input_width = input_cols.size();

	Dataset ds(train_row_nb, test_row_nb, total_input_width, total_output_width);

    ds.metadata.input_category_counts = std::move(in_cat_counts);
    ds.metadata.output_category_counts = std::move(out_cat_counts);

	//TRAINING DATA
    for (size_t r = 1; r <= train_row_nb; ++r) {
        //Inputs
        for (size_t i = 0; i < input_cols.size(); ++i) {
            ds.input_data(r - 1, i) = input_cols[i]->transform(data(r, input_cols[i]->get_index()));
        }
        //Outputs - for categorical columns one hot encode
        size_t c_out = 0; //index used to track the current output column, since one categorical column can take multiple output columns due to one-hot encoding
        for (size_t o = 0; o < output_cols.size(); ++o) {
            double val = output_cols[o]->transform(data(r, output_cols[o]->get_index()));

            if (auto cat_col = dynamic_cast<CategoricalColumn*>(output_cols[o])) {
                size_t width = cat_col->get_categories().size();
                for (size_t w = 0; w < width; ++w) ds.output_data(r - 1, c_out + w) = 0.0;
                if (val >= 0.0) ds.output_data(r - 1, c_out + static_cast<size_t>(val)) = 1.0;
                c_out += width;
            }
            else {
                ds.output_data(r - 1, c_out++) = val;
            }
        }
    }
	//TEST DATA
    for (size_t r = train_row_nb+1; r <= num_rows; ++r) {
        //Inputs
        size_t test_idx = r - train_row_nb - 1;
        for (size_t i = 0; i < input_cols.size(); ++i) {
            ds.test_inputs(test_idx, i) = input_cols[i]->transform(data(r, input_cols[i]->get_index()));
        }
		//Outputs - for categorical columns one hot encode
        size_t c_out = 0;
        for (size_t o = 0; o < output_cols.size(); ++o) {
            double val = output_cols[o]->transform(data(r, output_cols[o]->get_index()));

            if (auto cat_col = dynamic_cast<CategoricalColumn*>(output_cols[o])) {
                size_t width = cat_col->get_categories().size();
                for (size_t w = 0; w < width; ++w) ds.test_outputs(test_idx, c_out + w) = 0.0;
                if (val >= 0.0) ds.test_outputs(test_idx, c_out + static_cast<size_t>(val)) = 1.0;
                c_out += width;
            }
            else {
                ds.test_outputs(test_idx, c_out++) = val;
            }
        }
    }
    return ds;
}

std::vector<std::string> DataPreprocessor::get_target_cols_names() const {
    std::vector<std::string> names;
    names.reserve(columns.size());

    for (const auto& col : columns) {
        if (col->include_column && col->is_target_column) {
            names.push_back(col->name);
        }
    }
    return names;
}

std::vector<std::string> DataPreprocessor::inverse_transform_prediction(const Matrix& prediction_row) const {
    if (!fitted) {
        throw std::logic_error("DataPreprocessor::inverse_transform_prediction called before fit() or load()");
    }
    std::vector<std::string> results;
    size_t c_out = 0;

    for (const auto& col : columns) {
        if (!col->include_column || !col->is_target_column) continue;

        // If cateorical column: One-Hot Encoding -> ArgMax
        if (auto cat_col = dynamic_cast<CategoricalColumn*>(col.get())) {
            size_t width = cat_col->get_categories().size();

            if (c_out + width > prediction_row.get_columns_nb()) {
                throw std::runtime_error("CRITICAL ERROR: Neural network output size is too small!");
            }

            size_t best_idx = 0;
            double max_prob = -1e9;

            for (size_t w = 0; w < width; ++w) {
                double prob = prediction_row(0, c_out + w);
                if (prob > max_prob) {
                    max_prob = prob;
                    best_idx = w;
                }
            }

            results.push_back(col->inverse_transform(static_cast<double>(best_idx)));
            c_out += width;
        }
		//If numerical column: just inverse transform the single value
        else {
            if (c_out + 1 > prediction_row.get_columns_nb()) {
                throw std::runtime_error("CRITICAL ERROR: Neural network output size is too small!");
            }
            double val = prediction_row(0, c_out);
            results.push_back(col->inverse_transform(val));

            c_out += 1; 
        }
    }

    return results;
}

void DataPreprocessor::save(const std::string& filename) const {
    if (!fitted) {
        throw std::logic_error("DataPreprocessor::save called before fit() or load()");
    }
    json j;
    j["columns"] = json::array();

    for (const auto& col : columns) {
        j["columns"].push_back(col->serialize());
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("DataPreprocessor::save - Cannot open file: " + filename);
    }
    file << j.dump(4);
}

void DataPreprocessor::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("DataPreprocessor::load - Cannot open file: " + filename);
    }

    json j;
    file >> j;

    columns.clear();
    fitted = false;
    for (const auto& col_json : j.at("columns")) {
        std::string type = col_json.at("type").get<std::string>();
        size_t idx = col_json.at("index").get<size_t>();
        std::string name = col_json.at("name").get<std::string>();

        std::unique_ptr<Column> col;
        if (type == "Numerical") {
            col = std::make_unique<NumericalColumn>(idx, name);
        }
        else if (type == "Categorical") {
            col = std::make_unique<CategoricalColumn>(idx, name);
        }
        else {
            throw std::runtime_error("DataPreprocessor::load - Unknown column type!");
        }

        col->deserialize(col_json);
        columns.push_back(std::move(col));
    }
    fitted = true;
}