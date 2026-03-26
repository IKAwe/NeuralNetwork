#pragma once
#include <vector>
#include <string_view>

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

	void add_cell(std::string_view cell);
	void set_columns_nb(size_t c);
	void reserve_cells(size_t number_of_cells);
	std::vector<char>& get_buffer();
	void finalize();


	//Getters
	size_t get_rows_nb() const;
	size_t get_columns_nb() const;

	std::string_view operator()(size_t r, size_t c) const {
		return cells[r * columns + c];
	}
};