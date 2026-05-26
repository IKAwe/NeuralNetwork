#pragma once

#pragma once
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "loss_functions.h"

enum class FunctionsTypes { MSE };

class LossFuncMaker {
public:
    static inline std::map<std::string, FunctionsTypes> registry = {
        {"MSE", FunctionsTypes::MSE},

    };

    static std::vector<const char*> get_available_names() {
        std::vector<const char*> names;
        for (const auto& [name, type] : registry) {
            names.push_back(name.c_str());
        }
        return names;
    }

    static std::unique_ptr<Loss> create_by_name(std::string name) {
        FunctionsTypes type = registry.at(name);
        switch (type) {
        case FunctionsTypes::MSE:   return std::make_unique<MSE>();
        default: return nullptr;
        }
    }
};