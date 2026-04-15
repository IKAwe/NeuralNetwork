#include "csv_loader.h"
#include <fstream>
#include <filesystem>
#include <string_view>

/**
 * @brief Read a CSV file and load its content into a StringMatrix. The function reads the entire file into a buffer, then parses it to fill the cells of the matrix.
 * @param path The path to the CSV file to be loaded.
 * @return The StringMatrix containing the data from the CSV file. The matrix is filled with string_views that point to the corresponding substrings in the buffer.
 */
StringMatrix load_csv(const std::string& path) {
	StringMatrix matrix;

	//Open file and check if opened
	std::ifstream file(path, std::ios::binary);
	if(!file) {
        throw std::runtime_error("Could not open file: " + path);
	}

	//Get file size and reserve buffer size
	size_t file_size = static_cast<size_t>(std::filesystem::file_size(path));
	std::vector<char>& buffer = matrix.get_buffer();
	buffer.resize(file_size);

	//Read file content into buffer
    if (!file.read(buffer.data(), file_size)) {
        throw std::runtime_error("Error when trying to read file: " + path);
    }

	//Read file content into buffer
    size_t expected_col_nb = 0;
    size_t current_col_nb = 0;
    char* start = matrix.get_buffer().data();
	char* end = start + file_size;

    //For ever char 
    for (char* current = start; current <= end; ++current) {
        if (*current == ',' || *current == '\n' || current == end) {
            // trim \r
            size_t len = current - start;
            if (len > 0 && *(current - 1) == '\r') len--;

			// Add cell if came to a comma or newline
            matrix.add_cell(std::string_view(start, len));
            current_col_nb++;


            if (*current == '\n' || current == end) {
                if (expected_col_nb == 0) {
                    expected_col_nb = current_col_nb; // The first row defines the expected number of columns
                    matrix.set_columns_nb(expected_col_nb);

					// Estimate the number of rows based on the size of the second row and reserve space in the matrix accordingly
					char* char_in_second_row = current + 1; // Start of the second row
                    size_t estimated_row_size = 0;
					while (char_in_second_row < end && *char_in_second_row != '\n') {
						estimated_row_size++; 
						char_in_second_row++;
					}
					//Reserve space for estimated number of cells
                    if (estimated_row_size > 0) {
                        matrix.reserve_cells((file_size / estimated_row_size) * expected_col_nb); // Estimate number of rows and reserve space accordingly
                    }
                }
					
                else if (current_col_nb < expected_col_nb) {
					// Padding - if the current row has fewer columns than expected
                    for (size_t i = 0; i < (expected_col_nb - current_col_nb); ++i) {
                        matrix.add_cell(std::string_view(""));
                    }
                }
                current_col_nb = 0;
            }
            start = current + 1;
        }
    }

	// Handle the last cell if the file does not end with a newline
	//Check if our start pointer is still before the end of the buffer, which means there is a last cell to add
    if (start < end) {
        size_t len = end - start;
        if (len > 0 && *(end - 1) == '\r') len--;
        matrix.add_cell(std::string_view(start, len));
        current_col_nb++;

        // Padding
        if (expected_col_nb > 0 && current_col_nb < expected_col_nb) {
            for (size_t i = 0; i < (expected_col_nb - current_col_nb); ++i)
                matrix.add_cell(std::string_view(""));
        }
	}

	matrix.finalize(); // Finalize by calculating the number of rows based on the number of cells
    return matrix;//move instead of copy
}

