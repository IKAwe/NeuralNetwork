#pragma once
#include <string>
#include "data_structures.h"

std::vector<std::string> find_files_by_extension(const std::string& extension = ".csv", const std::string& folder = ".");
StringMatrix load_csv(const std::string& filename);