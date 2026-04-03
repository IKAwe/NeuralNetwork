#pragma once
#include <string>
#include <vector>
#include <memory>
#include "data_structures.h"
class Column {
public:
	std::string name;
	size_t index;
	bool include_column = false; //indicates whether the column is included in the input features or output labels.
	bool is_target_column = false;
	virtual ~Column() = default;
	virtual void fit(const StringMatrix& data)=0;
	virtual double transform(const std::string_view cell)=0;
	//Maybe add fit_and_transform functions that do both operations in one pass through the data, for efficiency.
};

class NumericalColumn : public Column {
	public:
	double mean;
	double std_dev;
	void fit(const StringMatrix & data) override;
	double transform(const std::string_view cell) override;
};
class CategoricalColumn : public Column {
	public:
	std::vector<std::string> categories;
	void fit(const StringMatrix& data) override;
	double transform(const std::string_view cell) override;
};
struct Dataset {
	Matrix input_data;
	Matrix output_data;
	Dataset(size_t row_nb, size_t input_col_nb, size_t output_col_nb) : input_data(row_nb, input_col_nb), output_data(row_nb, output_col_nb) {};
};

class DataPreprocessor {
	Matrix input_data, output_data;  
	std::vector<std::unique_ptr<Column>> columns;
public:
	/**
	 * @brief Adds all columns - based on name and index. The function should be called before fit() and transform() functions. Thank to this user will be able to specify which columns should be included, which are the target columns, which are numerical and which are categorical.
	 * @param data 
	 */
	void initialize_from_data(const StringMatrix& data);

	/**
	 * @brief Collect information from the data to be able to transform it later. For numerical columns - calculating the mean and standard deviation, 
	 for categorical columns - identifying the unique categories. This function should be called before transform() and should be called only once for a given dataset.
	 * @param data The data to fit the preprocessor on. 
	 */
	void fit(const StringMatrix& data);
	/**
	 * @brief Transform the input data into two matrices - one for input features and one for output labels. The transformation is done based on the information collected in the fit() function. 
	 For numerical columns, the data is normalized using the mean and standard deviation calculated in fit(). For categorical columns, the data is transformed into 
	 int value - the index of the cathegory based on the number of categories identified in fit().
	 * @param data The data to be transformed. Can be the same data that was used in fit() or new data with the same structure (same columns in the same order).
	 */
	Dataset transform(const StringMatrix& data);
	/**
	 * @brief Save the preprocessor state to a file. More precisely "columns" vector is saved - hence for numerical columns, mean and std_dev are saved, 
	 and for categorical columns, the list of categories is saved. This allows the preprocessor to be reloaded later and used to transform 
	 new data in the same way as the original data.
	 * @param filename The name of the file to save the preprocessor state to.
	 */
	void save(const std::string& filename) const;
	/**
	 * @brief Load the preprocessor state from a file. This should be used to load a preprocessor that was previously saved using the save() function.
	 * @param filename The name of the file to load the preprocessor state from. The file should contain the state of the preprocessor as saved by the save() function.
	 */
	void load(const std::string& filename);
};