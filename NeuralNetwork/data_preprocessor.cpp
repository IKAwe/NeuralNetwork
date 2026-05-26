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
/**
 * @brief Inverse transform a normalized value back to the original scale using the mean and standard deviation calculated in the fit() function.
 * @param val The normalized value to be inverse transformed.
 * @return The original value as a string.
 */
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

/**
 * @brief Collect information about the unique categories in the column. This is done by iterating through the column values, collecting them into a vector of string_views, sorting the vector to bring duplicates together, and then using std::unique to remove duplicates. The unique categories are then stored in the "categories" member variable for later use in the transform() function.
 * @param data The data to fit the preprocessor on. The function assumes that the first row of the data contains column headers and starts processing from the second row. The function also assumes that the column values are strings and does not perform any type checking or conversion.
 */
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
 * @brief Transform a cell value by finding its index in the categories vector. If the scale flag is set, the index is scaled to be between 0 and 1 based on the number of categories. If the category is not found in the categories vector, -1.0 is returned to indicate an unknown category.
 * @param cell The cell value to be transformed.
 * @return The transformed value as a double.
 */
double CategoricalColumn::transform(const std::string_view cell) {
	//Since categories are sorted, we can use binary search to find the index of the category
    auto it = std::lower_bound(categories.begin(), categories.end(), cell);

	// Check if the category was found and return its index as a double. If not found, return -1.0.
    if (it != categories.end() && *it == cell) {
        if (scale) {
            return static_cast<double>(std::distance(categories.begin(), it)) / (categories.size()-1);
        }
        else {
            return static_cast<double>(std::distance(categories.begin(), it));
        }
    }

    return -1.0;
}
/**
 * @brief Transform a normalized value back to the original category string. If the scale flag is set, the value is first denormalized to get the category index. If the index is out of bounds, "UNKNOWN_CATEGORY" is returned to indicate an unknown category.
 * @param val The normalized value to be inverse transformed.
 * @return The original category string.
 */
std::string CategoricalColumn::inverse_transform(double val) const {
    //Denormalise according to flag (we assume that range -0.5 to 0 is corresponding to 0th category)
    if (scale) {
        if (val <= -0.5) {
            return "UNKNOWN_CATEGORY";
        }
        else {
            val = std::clamp(val, 0.0, 1.0);
            val *= categories.size() -1;
        }
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
    j["scale"] = scale;
    return j;
}
void CategoricalColumn::deserialize(const json& j) {
    Column::deserialize(j);
    categories = j.at("categories").get<std::vector<std::string>>();
    scale = j.at("scale").get<bool>();
}


// --- DataPreprocessor Implementation ---

/**
* @brief Adds all columns - based on name and index. The function should be called before fit() and transform() functions. Thank to this user will be able to specify which columns should be included, which are the target columns, which are numerical and which are categorical.
* @param data The data to initialize the preprocessor from. The function assumes that the first row of the data contains column headers and starts processing from the second row. The type of each column is determined based on the value in the second row - if it can be converted to a double, it is considered numerical, otherwise it is considered categorical.
*/
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
/**
 * @brief Collect information from the data to be able to transform it later. For numerical columns - calculating the mean and standard deviation,
 for categorical columns - identifying the unique categories. This function should be called before transform() and should be called only once for a given dataset.
 * @param data The data to fit the preprocessor on.
 */
void DataPreprocessor::fit(const StringMatrix& data) {
    fitted = false;
    for (auto& col : columns) {
		if (col->include_column) col->fit(data);
    }
    fitted = true;
}
/**
 * @brief Transform the input data into two matrices - one for input features and one for output labels. The transformation is done based on the information collected in the fit() function.
 For numerical columns, the data is normalized using the mean and standard deviation calculated in fit(). For categorical columns, the data is transformed into
  value - the index of the cathegory based on the number of categories identified in fit() or scaled between -1 and 1 if the scale flag is set in Category class.
 * @param data The data to be transformed. Can be the same data that was used in fit() or new data with the same structure (same columns in the same order).
 */
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
        }
        else {
            input_cols.push_back(col.get());
            in_cat_counts.push_back(cat_count);
        }
    }

	Dataset ds(train_row_nb, test_row_nb, input_cols.size(), output_cols.size());

    ds.metadata.input_category_counts = std::move(in_cat_counts);
    ds.metadata.output_category_counts = std::move(out_cat_counts);

    for (size_t r = 1; r <= train_row_nb; ++r) {

        for (size_t i = 0; i < input_cols.size(); ++i) {
            ds.input_data(r - 1, i) = input_cols[i]->transform(data(r, input_cols[i]->get_index()));
        }
        for (size_t o = 0; o < output_cols.size(); ++o) {
            ds.output_data(r - 1, o) = output_cols[o]->transform(data(r, output_cols[o]->get_index()));
        }
    }
    for (size_t r = train_row_nb+1; r <= num_rows; ++r) {
        size_t test_idx = r - train_row_nb - 1;
        for (size_t i = 0; i < input_cols.size(); ++i) {
            ds.test_inputs(test_idx, i) = input_cols[i]->transform(data(r, input_cols[i]->get_index()));
        }
        for (size_t o = 0; o < output_cols.size(); ++o) {
            ds.test_outputs(test_idx, o) = output_cols[o]->transform(data(r, output_cols[o]->get_index()));
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
/**
 * @brief Get the original string values of the predicted output based on the transformed prediction row. This is done by applying the inverse_transform function of each target column to the corresponding value in the prediction row. The order of the values in the prediction row should correspond to the order of the target columns in the columns vector. This function is useful for interpreting the predictions made by the model in terms of the original data.
 * @param prediction_row The row of predictions to be inverse transformed. It should have the same number of columns as the number of target columns in the preprocessor. Each value in the row is expected to be a transformed value that can be inverse transformed using the corresponding target column's inverse_transform function.
 * @return The original string values of the predicted output. The order of the values in the returned vector corresponds to the order of the target columns in the columns vector.
 */
std::vector<std::string> DataPreprocessor::inverse_transform_prediction(const Matrix& prediction_row) const {
    if (!fitted) {
        throw std::logic_error("DataPreprocessor::inverse_transform_prediction called before fit() or load()");
    }
    std::vector<std::string> results;
    results.reserve(prediction_row.get_columns_nb());

    int pred_col_idx = 0;

    for (size_t i = 0; i < columns.size(); ++i) {
        auto& col = columns[i];

        if (col->include_column && col->is_target_column) {
            if (pred_col_idx < prediction_row.get_columns_nb()) {//maybe delete the condition
                results.push_back(col->inverse_transform(prediction_row(0, pred_col_idx)));
                pred_col_idx++;
            }
        }
    }
    return results;
}
/**
 * @brief Save the preprocessor state to a file. More precisely "columns" vector is saved - hence for numerical columns, mean and std_dev are saved,
 and for categorical columns, the list of categories is saved. This allows the preprocessor to be reloaded later and used to transform
 new data in the same way as the original data.
 * @param filename The name of the file to save the preprocessor state to.
 */
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
/**
 * @brief Load the preprocessor state from a file. This should be used to load a preprocessor that was previously saved using the save() function.
 * @param filename The name of the file to load the preprocessor state from. The file should contain the state of the preprocessor as saved by the save() function.
 */
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