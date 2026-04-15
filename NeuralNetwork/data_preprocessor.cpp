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
    return (val - mean) / std_dev;
}

void NumericalColumn::save(std::ostream& out) const {
    out << "0 " << name << " " << index << " "
        << is_target_column << " " << mean << " " << std_dev << "\n";
}
void NumericalColumn::load(std::istream& in){
    in >> is_target_column >> mean >> std_dev;
}

// --- CategoricalColumn Implementation ---


void CategoricalColumn::fit(const StringMatrix& data) {
	// Get all values as string_views to avoid unnecessary string constructions
    std::vector<std::string_view> raw_views;
    raw_views.reserve(data.get_rows_nb() - 1);

    for (size_t r = 1; r < data.get_rows_nb(); ++r) {
        raw_views.push_back(data(r, index));
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

double CategoricalColumn::transform(const std::string_view cell) {
	//Since categories are sorted, we can use binary search to find the index of the category
    auto it = std::lower_bound(categories.begin(), categories.end(), cell);

	// Check if the category was found and return its index as a double. If not found, return -1.0.
    if (it != categories.end() && *it == cell) {
        return static_cast<double>(std::distance(categories.begin(), it));
    }

    return -1.0;
}
void CategoricalColumn::save(std::ostream& out) const {
    out << "1 " << name << " " << index << " "
        << is_target_column << " " << categories.size();
    for (const auto& c : categories) {
        out << " " << c;
        out << "\n";
    }
}
void CategoricalColumn::load(std::istream& in) {
    size_t cat_count;
    in >> is_target_column >> cat_count;

    categories.clear();
    categories.reserve(cat_count);
    for (size_t i = 0; i < cat_count; ++i) {
        std::string cat_name;
        in >> cat_name;
        categories.push_back(cat_name);
    }
}

// --- DataPreprocessor Implementation ---
void DataPreprocessor::initialize_from_data(const StringMatrix& data) {
    columns.clear();
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
    for (auto& col : columns) {
		if (col->include_column) col->fit(data);
    }
}

Dataset DataPreprocessor::transform(const StringMatrix& data) {
    size_t num_rows = data.get_rows_nb() - 1;
    size_t input_cols_count = 0;
    size_t output_cols_count = 0;

	// Count how many input and output columns we have based on the is_target_column flag
    for (const auto& col : columns) {
		if (!col->include_column) continue; // Skip not included
        if (col->is_target_column) output_cols_count++;
        else input_cols_count++;
    }

	Dataset ds(num_rows, input_cols_count, output_cols_count);

	//Calculate category counts for metadata
	size_t meta_in_idx = 0; // To  prevent out of range access when some columns are not included or all are target columns
    size_t meta_out_idx = 0;

    for (const auto& col : columns) {
        if (!col->include_column) continue;

        int cat_count = 0;
        if (auto cat_col = dynamic_cast<CategoricalColumn*>(col.get())) {
            cat_count = static_cast<int>(cat_col->get_categories().size());
        }

        if (col->is_target_column) {
            // U¿ywamy meta_out_idx, a nie i!
            ds.metadata.output_category_counts[meta_out_idx++] = cat_count;
        }
        else {
            // U¿ywamy meta_in_idx, a nie i!
            ds.metadata.input_category_counts[meta_in_idx++] = cat_count;
        }
    }
	// Transform data row by row
    for (size_t r = 1; r < data.get_rows_nb(); ++r) {
        size_t in_idx = 0;
        size_t out_idx = 0;

        for (const auto& col : columns) {
			if (!col->include_column) continue; // Skip not included
            double val = col->transform(data(r, col->get_index()));

            if (col->is_target_column) {
                ds.output_data(r - 1, out_idx++) = val;
            }
            else {
                ds.input_data(r - 1, in_idx++) = val;
            }
        }
    }
    return ds;
}

void DataPreprocessor::save(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) return;

    file << columns.size() << "\n";
    for (const auto& col : columns) {
        col->save(file);
    }
}

void DataPreprocessor::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return;

    columns.clear();
    size_t col_count;
    file >> col_count;

    for (size_t i = 0; i < col_count; ++i) {
        int type;
        std::string name;
        size_t index;

        if (!(file >> type >> name >> index)) break;

        std::unique_ptr<Column> col;

        if (type == 0) {
            col = std::make_unique<NumericalColumn>(index, name);
        }
        else {
            col = std::make_unique<CategoricalColumn>(index, name);
        }

        col->load(file);

        columns.push_back(std::move(col));
    }
}