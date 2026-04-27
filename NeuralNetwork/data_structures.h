#pragma once
#include <vector>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <iomanip>

/**
 * @brief For usage especially when we dont know the number of rows. First - add al the cells with add_cell, then set the number of columns with set_columns_nb, 
 and finally call finalize to calculate the number of rows based on the number of cells and columns.
 */
class StringMatrix {
	std::vector<char> buffer;
	std::vector<std::string_view> cells;

	size_t rows = 0;
	size_t columns = 0;
public:
	StringMatrix(size_t r, size_t c) : rows(r), columns(c) {
		cells.resize(r * c);
		buffer.reserve(r * c * 10); // Assuming average cell length of 10 characters, adjust as needed
	};
	StringMatrix() = default;
	/**
 * @brief Add a string_view cell to cells vector. The string_view should point to a valid string in the buffer vector.
 * @param cell string_view to be added to cells vector.
 */
	void add_cell(std::string_view cell) {
		cells.push_back(cell);
	};
	/**
 * @brief Define the number of columns in the matrix.
 * @param c The number of columns in the matrix.
 */
	void set_columns_nb(size_t c) {
		columns = c;
	};
	/**
 * @brief Reserve space in the cells vector for a given number of rows and columns. This is an optimization to avoid multiple reallocations as cells are added to the matrix.
 * @param r Rows to reserve space for.
 * @param c Columns to reserve space for.
 */
	void reserve_cells(size_t number_of_cells) {
		cells.reserve(number_of_cells);
	};
	/**
 * @brief
 * @return A reference to the buffer vector.
 */
	std::vector<char>& get_buffer() { return buffer; };

	/**
	 * @brief Finalize the matrix by calculating the number of rows based on the number of cells and columns. This should be called after all cells have been added and the number of columns has been set.
	 */
	void finalize() { rows = cells.size() / columns; };


	//Getters
	size_t get_rows_nb() const {
		return rows;
	};
	size_t get_columns_nb() const {
		return columns;
	};
	//For writing to the matrix
	std::string_view& operator()(size_t r, size_t c) {
		return cells[r * columns + c];
	}
	//For reading from the matrix
	std::string_view operator()(size_t r, size_t c) const {
		return cells[r * columns + c];
	}
};

class Matrix{
	std::vector<double> data;
	size_t rows_nb = 0;
	size_t col_nb = 0;
public:
	//Matrix() = default;
	Matrix(size_t r, size_t c) : rows_nb(r), col_nb(c), data(r* c, 0.0) {};
	//for saving
	double& operator()(size_t r, size_t c) {
		if (r >= rows_nb || c >= col_nb) {
			throw std::out_of_range("Matrix index out of range");
		}
		return data[r * col_nb + c];
	}
	//for reading
	double operator()(size_t r, size_t c) const {
		if (r >= rows_nb || c >= col_nb) {
			throw std::out_of_range("Matrix index out of range");
		}
		return data[r * col_nb + c];
	}
	//Used for X*W
	Matrix operator*(const Matrix& other) const {
		if (col_nb != other.rows_nb) {
			throw std::runtime_error("Matrix dimensions do not match for multiplication");
		}
		Matrix result(rows_nb, other.col_nb);
		for (size_t i = 0; i < rows_nb; ++i) {
			for (size_t k = 0; k < col_nb; ++k) {
				for (size_t j = 0; j < other.col_nb; ++j) {
					result(i, j) += (*this)(i, k) * other(k, j);
				}
			}
		}
		return result;
	}
	Matrix transpose() const {
		Matrix result(col_nb, rows_nb);
		for (size_t r = 0; r < rows_nb; ++r) {
			for (size_t c = 0; c < col_nb; ++c) {
				result(c, r) = (*this)(r, c);
			}
		}
		return result;
	}
	Matrix hadamard(const Matrix& other) const {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Dimensions must match for Hadamard product");
		}
		Matrix result(rows_nb, col_nb);
		for (size_t i = 0; i < data.size(); ++i) {
			result.data[i] = data[i] * other.data[i];
		}
		return result;
	}

	Matrix& operator+=(const Matrix& other) {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Dimensions do not match for addition");
		}
		for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
		return *this;
	}
	void zero() {
		std::fill(data.begin(), data.end(), 0.0);
	}
	void resize(size_t new_rows, size_t new_cols) {
		rows_nb = new_rows;
		col_nb = new_cols;
		data.resize(new_rows * new_cols, 0.0);
	}
	void print() const {
		for (size_t r = 0; r < rows_nb; ++r) {
			std::cout << "[";
			for (size_t c = 0; c < col_nb; ++c) {
				std::cout << std::setw(6) << std::right << (*this)(r, c) << " ";
			}
			std::cout << "]" << std::endl;
		}
	}
	 const std::vector<double>& get_data() const {
		return data;
	}
	 std::vector<double>& get_data_mutable() {
		 return data;
	 }

	size_t get_rows_nb() const { return rows_nb; }
	size_t get_columns_nb() const { return col_nb; }
};