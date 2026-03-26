#include "data_structures.h"

/**
 * @brief Add a string_view cell to cells vector. The string_view should point to a valid string in the buffer vector.
 * @param cell string_view to be added to cells vector.
 */
void StringMatrix::add_cell(std::string_view cell) {
	cells.push_back(cell);
}

/**
 * @brief Define the number of columns in the matrix.
 * @param c The number of columns in the matrix.
 */
void StringMatrix::set_columns_nb(size_t c) {
	columns = c;
}

/**
 * @brief Reserve space in the cells vector for a given number of rows and columns. This is an optimization to avoid multiple reallocations as cells are added to the matrix.
 * @param r Rows to reserve space for.
 * @param c Columns to reserve space for.
 */
void StringMatrix::reserve_cells(size_t number_of_cells) {
	cells.reserve(number_of_cells);
}

/**
 * @brief 
 * @return A reference to the buffer vector.
 */
std::vector<char>& StringMatrix::get_buffer() { return buffer; }

/**
 * @brief Finalize the matrix by calculating the number of rows based on the number of cells and columns. This should be called after all cells have been added and the number of columns has been set.
 */
void StringMatrix::finalize() { rows = cells.size() / columns; }


//Getters
size_t StringMatrix::get_rows_nb() const {
	return rows;
}
size_t StringMatrix::get_columns_nb() const {
	return columns;
}