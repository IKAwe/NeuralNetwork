#pragma once
#include <vector>
#include <string_view>
#include <stdexcept>

/**
 * @brief 
 */
class StringMatrix {
	std::vector<char> buffer;
	std::vector<std::string_view> cells;

	size_t rows = 0;
	size_t columns = 0;
public:
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

	std::string_view operator()(size_t r, size_t c) const {
		return cells[r * columns + c];
	}
};

class Matrix{
	std::vector<double> data;
	size_t rows_nb = 0;
	size_t col_nb = 0;
public:

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
	size_t get_rows() const { return rows_nb; }
	size_t get_cols() const { return col_nb; }
};