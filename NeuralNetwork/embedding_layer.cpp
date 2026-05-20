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

bool TabularEmbeddingLayer::initialize() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.01);

    for (auto& table : embedding_tables) {
        for (size_t r = 0; r < table.get_rows_nb(); ++r) {
            for (size_t c = 0; c < table.get_columns_nb(); ++c) {
                table(r, c) = dist(gen);
            }
        }
    }
    return true;
}

Matrix TabularEmbeddingLayer::feedforward(const Matrix& inputs) {
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
    size_t num_tables = embedding_tables.size();
    out.write(reinterpret_cast<const char*>(&num_tables), sizeof(num_tables));
    for (const auto& table : embedding_tables) {
        table.save(out);
    }
}

void TabularEmbeddingLayer::load(std::istream& in) {
    size_t num_tables;
    in.read(reinterpret_cast<char*>(&num_tables), sizeof(num_tables));
    if (num_tables == embedding_tables.size()) {
        for (auto& table : embedding_tables) {
            table.load(in);
        }
    }
}