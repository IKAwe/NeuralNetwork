# C++ Neural Engine & GUI

A Multilayer Perceptron neural network written entirely from scratch in C++. I built this engine to train models without relying on heavy, black-box frameworks like PyTorch or TensorFlow, while keeping a live, interactive view of the data and training process.

The focus here is on low-level memory optimization and a completely unblocked UI.

## Key Features

* **Zero-Allocation Training Loop:** Mini-batch training runs without a single heap allocation inside the main loop. Data is overwritten in pre-allocated matrix buffers on the fly. This heavily reduces CPU overhead and keeps the CPU cache happy.
* **Interactive GUI:** 
  * Tweak the network architecture on the fly (add hidden layers, change activations, set targets).
  * Asynchronous logs and Loss charting. The training loop runs on a dedicated background thread, so the UI never freezes (thread-safe communication via Mutexes).
* **Standalone Data Preprocessor:** A dedicated module for parsing raw CSV inputs. Automatically scales numerical features (Mean / Std Dev) and handles categorical data encoding. 
* **State Management:** Full save/load support. Preprocessor configs are saved as readable `.json` files, while trained network weights are exported directly to fast `.bin` binaries.

## Tech Stack

* **Language:** C++ (Standard 17+)
* **GUI:** [Dear ImGui](https://github.com/ocornut/imgui) + GLFW + OpenGL3
* **Parsing:** [nlohmann/json](https://github.com/nlohmann/json) (Header-only)
* **Math:** Custom `Matrix` class implementation built for speed (SIMD/CUDA).

## GUI Preview
<img width="1117" height="601" alt="image" src="https://github.com/user-attachments/assets/48686120-d77a-4d6f-8f56-8fb0a93a3725" />

##  How to Build & Run

This project is set up as a Visual Studio Solution.

1. Clone the repo.
2. Make sure you have the dependencies linked (`ImGui`, `GLFW`, and `nlohmann/json`). 
3. Open the `.sln` file in Visual Studio.
4. **Crucial:** Switch the build configuration to **Release / x64**. (If you run the training loop in Debug mode, it will be slow).
5. Build (`Ctrl + Shift + B`) and run.

##  Under the Hood

A quick look at the engine's architecture:
* `DataPreprocessor` - Parses CSVs, generates metadata, standardizes features, and outputs a clean `Dataset`.
* `NeuralNetwork` - The core engine. Handles the Forward Pass, calculates Loss (e.g., MSE), and runs Backpropagation.
* `Layer` (Dense, Activations, Dropout) - Polymorphic layer design where each layer handles its own gradient math.
* `GUI` - Strictly separated view layer. Talks to the engine using safe callbacks (like `EpochStats`).

---
*Built as a deep dive into the math behind neural networks (backprop from scratch) and C++ optimization.*
