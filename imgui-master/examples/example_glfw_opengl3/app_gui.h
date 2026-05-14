#pragma once
#include <string>
#include <vector>
#include "neural_network.h"
#include "layer_maker.h"
#include "data_preprocessor.h"
#include "loss_func_maker.h"
#include <optional>
#include <thread>
#include <atomic>
#include <mutex>

struct AppState {
    //Choosing a file
    std::vector<std::string> csv_files;
    int selected_file_idx = 0;
    //Datapreprocessing
    DataPreprocessor preprocessor;
    std::atomic<bool> is_fitting = false;
    bool is_fitted = false;
    StringMatrix raw_data;
    std::atomic<bool> is_transforming = false;
    std::optional<Dataset> dataset;
    //Architecture
    std::vector<LayerUI> gui_layers;
    std::vector<const char*> layer_names = LayerMaker::get_available_names();
    // --- Stan dla zak³adki TRAIN ---
    NeuralNetwork nn;
    Hyperparams hyperparams = Hyperparams(100, 32, 0.5f);
    std::vector<const char*> loss_names = LossFuncMaker::get_available_names();
    int selected_loss_idx = 0;
    std::atomic<bool> is_training = false;

    std::vector<float> loss_history;  //do zminny na matrix
    std::vector<std::string> training_logs;
    std::mutex gui_mutex; // Do synchronizacji dostêpu do loss_history i training_logs

    // --- Stan dla zak³adki PREDICT ---
    char output_filename[128] = "predictions.csv";
    std::string selected_model = "Brak modelu";
    std::string selected_data_predict = "Brak danych";

};

class AppGUI {
private:
    
    AppState state;
    void renderTrainTab();
    void renderPredictTab();

public:
    AppGUI() = default;
    void render(); // G³ówna funkcja wywo³ywana co klatkê
};
