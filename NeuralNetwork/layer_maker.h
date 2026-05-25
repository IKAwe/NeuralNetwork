#pragma once
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "layers.h"
#include "activation_functions.h"
#include "embedding_layer.h"

enum class LayerType { Dense, ReLU, Sigmoid, Dropout, TabularEmbedding, Tanh, LeakyReLU };

struct LayerUI {
    int type_index = 0; 

    int inputs = 10;
    int outputs = 10;

    float dropout_rate = 0.5f;
};

class LayerMaker {
public:
    static inline std::map<std::string, LayerType> registry = {
        {"Dense", LayerType::Dense},
       {"ReLU", LayerType::ReLU},
        {"Sigmoid", LayerType::Sigmoid},
		{"TabularEmbedding", LayerType::TabularEmbedding},
        {"Tanh", LayerType::Tanh},
		{"LeakyReLU", LayerType::LeakyReLU}
        //{"Dropout", LayerType::Dropout}
    };

    static std::vector<const char*> get_available_names() {
        std::vector<const char*> names;
        for (const auto& [name, type] : registry) {
            names.push_back(name.c_str());
        }
        return names;
    }

    // 3. Tworzenie warstwy na podstawie nazwy lub typu
    static std::unique_ptr<Layer> create_by_name(std::string name, size_t id, const LayerUI& config) {
        LayerType type = registry.at(name);
        switch (type) {
        case LayerType::Dense:   return std::make_unique<Dense>(id, config.inputs, config.outputs);
        case LayerType::ReLU:    return std::make_unique<ReLU>(id, config.inputs);
        case LayerType::Sigmoid: return std::make_unique<Sigmoid>(id, config.inputs);
		case LayerType::Tanh:    return std::make_unique<Tanh>(id, config.inputs);
		case LayerType::LeakyReLU:return std::make_unique<LeakyReLU>(id, config.inputs, 0.01);
        case LayerType::TabularEmbedding: return std::make_unique<TabularEmbeddingLayer>(id, config.inputs, config.outputs);
        //case LayerType::Dropout: return std::make_unique<Dropout>(id, in, rate);
        default: return nullptr;
        }
    }
};