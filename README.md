# C++ Neural Engine & GUI

A Multilayer Perceptron neural network written from scratch in C++. I built this engine to train models without relying on heavy, black-box frameworks like PyTorch or TensorFlow, while keeping a live, interactive view of the data and training process.

The focus here is on low-level memory optimization and a completely unblocked UI.
## Branches
This repository is split into specific branches based on the features you need:
* **`master`**: The lightweight, core version of the engine. Supports standard Dense layers, Dropout, Regression (MSE), and basic Label Encoding.
* **`accelerated`**: A fully GPU-accelerated version of the matrix math engine using cuBLAS.
* **`embeddings-onehot_output`**: The advanced version. Introduces **Tabular Embedding Layers** for high-cardinality categorical inputs, One-Hot Encoding (OHE) for targets. Best for datasets with many categorical columns. Uses accelerations from 'accelerated' branch.

## Key Features

* **Zero-Allocation Training Loop:** Mini-batch training runs without a single heap allocation inside the main loop. Data is overwritten in pre-allocated matrix buffers on the fly. This heavily reduces CPU overhead and keeps the CPU cache happy.
* **Interactive GUI:** 
  * Tweak the network architecture on the fly (add hidden layers, change activations, set targets).
* **Standalone Data Preprocessor:** A dedicated module for parsing raw CSV inputs. Automatically scales numerical features and handles categorical data encoding. 
* **State Management:** Full save/load support. Preprocessor configs are saved as readable `.json` files, while trained network weights are exported directly to `.bin`s.

## Tech Stack

* **Language:** C++ (Standard 17+)
* **GUI:** [Dear ImGui](https://github.com/ocornut/imgui) + GLFW + OpenGL3
* **Parsing:** [nlohmann/json](https://github.com/nlohmann/json) (Header-only)
* **Math:** Custom `Matrix` class implementation built for speed (SIMD/CUDA).

## GUI Preview
<img width="1181" height="646" alt="image" src="https://github.com/user-attachments/assets/60c1012c-9929-469d-b30b-49b39e54a262" />
<img width="1181" height="651" alt="image" src="https://github.com/user-attachments/assets/0910c6e0-b227-4cc4-b9f7-59b5a8d56c6a" />


##  How to Build & Run

This project is set up as a Visual Studio Solution.

1. Clone the repo.
2. Make sure you have the dependencies linked (`ImGui`, `GLFW`, and `nlohmann/json`). 
3. Open the `.sln` file in Visual Studio.
4. **Crucial:** Switch the build configuration to **Release / x64**. (If you run the training loop in Debug mode, it will be slower).

## GPU Acceleration

A fully GPU-accelerated version of the matrix math engine (using cuBLAS) is available on a separate branch. 

To run the CUDA version:
1. Switch to the `accelerated` branch.
2. Ensure you have an NVIDIA GPU and the official [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed on your system.
3. Make sure the `USE_CUDA` preprocessor macro is defined in your Visual Studio project properties (or passed as an environment flag if building from CLI).
6. Build (`Ctrl + Shift + B`) and run.
Remember the accelerated version will work best for large datasets with large batch sizes(128+) and neurons per layer - otherwise it probably will be slower than not cuda-accelerated version.

---
*Built as a deep dive into the math behind neural networks and C++ optimization.*
