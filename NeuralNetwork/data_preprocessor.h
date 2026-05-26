#pragma once
#include <string>
#include <vector>
#include <memory>
#include "data_structures.h"
#include "json.hpp"

using json = nlohmann::json;

class Column {
public:
	std::string name;
	bool include_column = true; //indicates whether the column is included in the input features or output labels.
	bool is_target_column = false;

	virtual void fit(const StringMatrix& data)=0;
	virtual double transform(const std::string_view cell)=0;
	virtual std::string inverse_transform(const double value) const = 0;

	virtual json serialize() const {
		json j;
		j["index"] = index;
		j["name"] = name;
		j["include_column"] = include_column;
		j["is_target_column"] = is_target_column;
		return j;
	};
	virtual void deserialize(const json& j) {
		index = j.at("index").get<size_t>();
		name = j.at("name").get<std::string>();
		include_column = j.at("include_column").get<bool>();
		is_target_column = j.at("is_target_column").get<bool>();
	};

	virtual ~Column() = default;
protected:
	size_t index;
	Column(size_t idx, std::string name) : index(idx), name(std::move(name)) {};
public:
	size_t get_index() const { return index; };
};

class NumericalColumn : public Column {
	double mean = 0.0;
	double std_dev = 0.0;
public:
	double get_mean() const { return mean; };
	double get_std_dev() const { return std_dev; };

	void fit(const StringMatrix & data) override;
	double transform(const std::string_view cell) override;
	std::string inverse_transform(const double value) const override;
	json serialize() const override;
	void deserialize(const json& j) override;
	NumericalColumn(size_t idx, std::string name) : Column(idx, std::move(name)) {};
};
class CategoricalColumn : public Column {
	std::vector<std::string> categories;
	bool scale = true; //If true transform output will be between -1 and 1, else the output will be equal to category index
public:
	void fit(const StringMatrix& data) override;
	double transform(const std::string_view cell) override;
	std::string inverse_transform(const double value) const override;
	json serialize() const override;
	void deserialize(const json& j) override;
	const std::vector<std::string>& get_categories() const { return categories; };
	CategoricalColumn(size_t idx, std::string name) : Column(idx, std::move(name)) {};
};

struct DatasetMetadata {
	std::vector<int> input_category_counts; // For categorical columns, the number of unique categories(0 for numerical). This is needed to know how to implement embedding layer.
	std::vector<int> output_category_counts; 
	DatasetMetadata(size_t input_column_nb, size_t output_column_nb) : input_category_counts(input_column_nb), output_category_counts(output_column_nb) {};
};

struct Dataset {
	Matrix input_data;
	Matrix output_data;

	Matrix test_inputs;
	Matrix test_outputs;

	DatasetMetadata metadata;
	Dataset(size_t train_row_nb, size_t test_row_nb, size_t input_col_nb, size_t output_col_nb) : input_data(train_row_nb, input_col_nb), output_data(train_row_nb, output_col_nb), 
		test_inputs(test_row_nb, input_col_nb), test_outputs(test_row_nb, output_col_nb), metadata(input_col_nb, output_col_nb) { };
};

class DataPreprocessor {
	std::vector<std::unique_ptr<Column>> columns;
	bool fitted = false;
public:

	void initialize_from_data(const StringMatrix& data);
	const std::vector<std::unique_ptr<Column>>& get_columns() { return columns; };
	std::vector<std::unique_ptr<Column>>& get_columns_mutable() { return columns; };

	void fit(const StringMatrix& data);
	Dataset transform(const StringMatrix& data, double test_fraction = 0.2);

	//For prediction interpretation
	std::vector<std::string> get_target_cols_names() const;
	std::vector<std::string> inverse_transform_prediction(const Matrix& prediction_row) const;
	void save(const std::string& filename) const;

	void load(const std::string& filename);

	bool is_fitted() { return fitted; };
};