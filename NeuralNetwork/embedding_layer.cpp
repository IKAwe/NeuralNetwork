#include "embedding_layer.h"

TabularEmbeddingLayer::TabularEmbeddingLayer(size_t id, size_t input_cols, const std::vector<EmbeddingConfig>& confs)
    : Layer(id), configs(confs)
{
    this->input_nb = input_cols;

    size_t total_out = input_cols;
    for (const auto& conf : configs) {
        total_out += (conf.embed_dim - 1);
    }
    this->output_nb = total_out;

    for (const auto& conf : configs) {
        embedding_tables.push_back(Matrix(conf.vocab_size, conf.embed_dim));
        embedding_gradients.push_back(Matrix(conf.vocab_size, conf.embed_dim));
    }
}

TabularEmbeddingLayer::TabularEmbeddingLayer(size_t id, size_t input_cols, size_t output_cols)
    : Layer(id, input_cols, output_cols){}

bool TabularEmbeddingLayer::initialize() {
    if (embedding_tables.empty()) {
		throw std::runtime_error("TabularEmbeddingLayer::initialize called but no embedding tables were created.");
    }
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dist(0.0, 0.1);

    for (auto& table : embedding_tables) {
        size_t cols = table.get_columns_nb();
        if (cols == 0) continue;

        for (size_t r = 0; r < table.get_rows_nb(); ++r) {
            for (size_t c = 0; c < cols; ++c) {
                table(r, c) = dist(gen);
            }
        }
    }
    return true;
}

Matrix TabularEmbeddingLayer::feedforward(const Matrix& inputs) {
    if(embedding_tables.empty()) {
        throw std::runtime_error("TabularEmbeddingLayer::feedforward called but no embedding tables were created.");
	}

    size_t batch_size = inputs.get_rows_nb();
    Matrix output(batch_size, this->output_nb);

    for (size_t r = 0; r < batch_size; ++r) {
        size_t out_col_idx = 0;

        for (size_t in_col_idx = 0; in_col_idx < this->input_nb; ++in_col_idx) {
            auto it = std::find_if(configs.begin(), configs.end(),
                [in_col_idx](const EmbeddingConfig& c) { return c.input_col_index == in_col_idx; });

            if (it != configs.end()) {
                size_t config_idx = std::distance(configs.begin(), it);
                int cat_index = static_cast<int>(std::round(inputs(r, in_col_idx)));

                if (cat_index < 0 || cat_index >= it->vocab_size) cat_index = 0;

                for (size_t d = 0; d < it->embed_dim; ++d) {
                    output(r, out_col_idx++) = embedding_tables[config_idx](cat_index, d);
                }
            }
            else {
                output(r, out_col_idx++) = inputs(r, in_col_idx);
            }
        }
    }
    return output;
}
/**
 * @brief 
 * @param inputs 
 * @param gradients_from_next_layer 
 * @return Matrix filled with zeros.
 */
Matrix TabularEmbeddingLayer::backpropagate(const Matrix& inputs, const Matrix& gradients_from_next_layer) {
    if (embedding_tables.empty()) {
        throw std::runtime_error("TabularEmbeddingLayer::backpropagate called but no embedding tables were created.");
    }
    size_t batch_size = gradients_from_next_layer.get_rows_nb();

    Matrix input_gradients(batch_size, this->input_nb);

    //For every record
    for (size_t r = 0; r < batch_size; ++r) {
        size_t out_col_idx = 0;
        //for every input column
        for (size_t in_col_idx = 0; in_col_idx < this->input_nb; ++in_col_idx) {
            //find - if it exists - the columns embedding config
            auto it = std::find_if(configs.begin(), configs.end(),
                [in_col_idx](const EmbeddingConfig& c) { return c.input_col_index == in_col_idx; });
            //if it exists
            if (it != configs.end()) {
                size_t config_idx = std::distance(configs.begin(), it);
                int cat_index = static_cast<int>(std::round(inputs(r, in_col_idx)));
                if (cat_index < 0 || cat_index >= it->vocab_size) cat_index = 0;

                for (size_t d = 0; d < it->embed_dim; ++d) {
                    embedding_gradients[config_idx](cat_index, d) += gradients_from_next_layer(r, out_col_idx++);
                }
            }
            else {
                out_col_idx++;
            }
        }
    }

    return input_gradients; //Return empty Matrix because Embedding layer must be the first one
}

void TabularEmbeddingLayer::zero_gradients() {
    for (auto& grad_table : embedding_gradients) {
        grad_table.zero();
    }
}

void TabularEmbeddingLayer::update_params(double lr, size_t batch_size) {
    double scale = lr / static_cast<double>(batch_size);

    for (size_t i = 0; i < embedding_tables.size(); ++i) {
        embedding_tables[i] = embedding_tables[i] - (embedding_gradients[i] * scale);
    }
}

void TabularEmbeddingLayer::save(std::ostream& out) const {
    if (embedding_tables.empty()) {
        throw std::runtime_error("TabularEmbeddingLayer::save method called but no embedding tables were created.");
    }

    // 1. Save the number of configurations/tables
    size_t num_configs = configs.size();
    out.write(reinterpret_cast<const char*>(&num_configs), sizeof(num_configs));

    // 2. Save the details of each EmbeddingConfig
    for (const auto& conf : configs) {
        out.write(reinterpret_cast<const char*>(&conf.input_col_index), sizeof(conf.input_col_index));
        out.write(reinterpret_cast<const char*>(&conf.vocab_size), sizeof(conf.vocab_size));
        out.write(reinterpret_cast<const char*>(&conf.embed_dim), sizeof(conf.embed_dim));
    }

    // 3. Save the actual Embedding Matrices
    for (const auto& table : embedding_tables) {
        table.save(out);
    }
}

void TabularEmbeddingLayer::load(std::istream& in) {
    embedding_tables.clear();
    embedding_gradients.clear();
    configs.clear();

    // 1. Read the number of configurations/tables
    size_t num_configs;
    in.read(reinterpret_cast<char*>(&num_configs), sizeof(num_configs));

    size_t expected_output_nb = this->input_nb;

    // 2. Read the EmbeddingConfig details
    for (size_t i = 0; i < num_configs; ++i) {
        EmbeddingConfig conf;
        in.read(reinterpret_cast<char*>(&conf.input_col_index), sizeof(conf.input_col_index));
        in.read(reinterpret_cast<char*>(&conf.vocab_size), sizeof(conf.vocab_size));
        in.read(reinterpret_cast<char*>(&conf.embed_dim), sizeof(conf.embed_dim));

        configs.push_back(conf);
        expected_output_nb += (conf.embed_dim - 1);
    }

    // 3. Read the Embedding Matrices
    for (size_t i = 0; i < num_configs; ++i) {
        Matrix table(0, 0); // Dimensions will be overwritten by table.load()
        table.load(in);

        // Sanity Check: Ensure the loaded matrix matches the config
        if (table.get_rows_nb() != configs[i].vocab_size || table.get_columns_nb() != configs[i].embed_dim) {
            throw std::runtime_error("TabularEmbeddingLayer::load - Matrix dimensions do not match config.");
        }

        embedding_tables.push_back(table);
        // Initialize an empty gradient matrix of the exact same size
        embedding_gradients.push_back(Matrix(table.get_rows_nb(), table.get_columns_nb()));
    }

    // 4. Validate total dimensions
    if (this->output_nb != expected_output_nb) {
        throw std::runtime_error("TabularEmbeddingLayer::load - Calculated output dimension mismatch.");
    }
}