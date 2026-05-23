#pragma once
#include <string>
#include <vector>
#include "neural_network.h"
#include "layer_maker.h"
#include "embedding_layer.h"
#include "data_preprocessor.h"
#include "loss_func_maker.h"
#include <optional>
#include <thread>
#include <atomic>
#include <mutex>

struct AppState {
    //Choosing a file
    std::vector<std::string> csv_files;
    bool is_loading_csv = false;
    int selected_file_idx = -1; //Change to -1
    //Datapreprocessing
    DataPreprocessor preprocessor;

    char preprocessor_filepath[128] = "preprocessor_config.json";
    std::string preprocessor_status_msg = "";

    std::atomic<bool> is_fitting = false;
    StringMatrix raw_data;
    std::atomic<bool> is_transforming = false;
    std::optional<Dataset> dataset;

    //Architecture
    std::vector<LayerUI> gui_layers;
    std::vector<const char*> layer_names = LayerMaker::get_available_names();
    std::vector<EmbeddingConfig> embed_configs; // Used for creating embedding layer
    // --- Stan dla zak³adki TRAIN ---
    NeuralNetwork nn;
    char network_filepath[128] = "neural_network.bin";
    std::string network_status_msg = "";
    bool is_architecture_locked = false;

    Hyperparams hyperparams = Hyperparams(100, 32, 0.5f);
    std::vector<const char*> loss_names = LossFuncMaker::get_available_names();
    int selected_loss_idx = 0;
    std::atomic<bool> is_training = false;
    std::atomic<bool> stop_training = false;

    std::vector<float> loss_history;  //do zminny na matrix
    std::vector<float> test_loss_history;
    std::vector<std::string> training_logs;
    std::mutex gui_mutex; // Do synchronizacji dostêpu do loss_history i training_logs

    // --- Stan dla zak³adki PREDICT ---

    std::vector<std::string> bin_files;
    std::vector<std::string> json_files;
    //csv_files is already defined

    int selected_model_idx = -1;
    int selected_json_idx = -1;

    NeuralNetwork prediction_nn;
    DataPreprocessor prediction_preprocessor;

    std::map<size_t, float> predict_num_inputs;
    std::map<size_t, int> predict_cat_inputs;
    std::vector<std::string> predict_results;

    std::string predict_status_msg = "";

};

class AppGUI {
private:
    
    AppState state;
    void renderTrainTab();
    void renderPredictTab();

public:
    AppGUI() = default;
    void render();
};
