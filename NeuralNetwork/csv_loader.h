#pragma once
#include <string>
#include "data_structures.h"

std::vector<std::string> find_csv_files(const std::string& folder = ".");
StringMatrix load_csv(const std::string& filename);