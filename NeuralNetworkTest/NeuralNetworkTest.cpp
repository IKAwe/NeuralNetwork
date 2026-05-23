#include "pch.h"
#include <fstream>
#include "CppUnitTest.h"
#include "csv_loader.h"
#include "data_structures.h"
#include "layers.h"
#include "data_preprocessor.h"
#include "neural_network.h"
#include "loss_functions.h"
#include "embedding_layer.h"
#include "layer_maker.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkTest
{
	TEST_CLASS(CSVLoadingTests)
	{
    public:
        TEST_METHOD(Should_Parse_Entire_Matrix) {
            // 1. Arrange
            std::string test_data = "col1,col2,col3\n1,2,3";
            std::string path = "unit_test.csv";
            {
                std::ofstream f(path);
                f << test_data;
            } //close file

            // 2. Act 
            auto matrix = load_csv(path);
			std::remove(path.c_str());
            // 3. Assert 
			// Check the dimensions of the matrix
            Assert::AreEqual(size_t(3), matrix.get_columns_nb(), L"Number of columns should be 3");
            Assert::AreEqual(size_t(2), matrix.get_rows_nb(), L"Number of rows should be 2");

			// Check the content of the matrix

            // Header 
            Assert::AreEqual("col1", std::string(matrix(0, 0)).c_str());
            Assert::AreEqual("col2", std::string(matrix(0, 1)).c_str());
            Assert::AreEqual("col3", std::string(matrix(0, 2)).c_str());

            // Data
            Assert::AreEqual("1", std::string(matrix(1, 0)).c_str());
            Assert::AreEqual("2", std::string(matrix(1, 1)).c_str());
            Assert::AreEqual("3", std::string(matrix(1, 2)).c_str());
        }

        TEST_METHOD(Handle_Empty_Cells_Test) {
            std::string path = "empty_cells.csv";
            std::ofstream f(path);
			f << "col1,col2,col3\n1,,3\n,5,6\n7,\n";
            f.close();

            auto matrix = load_csv(path);

			std::remove(path.c_str());

			Assert::IsTrue(std::string(matrix(1, 1)).empty(), L"Empty cell should be an empty string");
			Assert::IsTrue(std::string(matrix(2, 0)).empty(), L"The padding should be added");
			Assert::IsTrue(std::string(matrix(3, 2)).empty(), L"The padding should be added");

		}

        TEST_METHOD(Should_Handle_Windows_Endings) {
            std::string path = "win.csv";
            std::ofstream f(path, std::ios::binary);
            f << "col1,col2\r\n1,2\r\n";
            f.close();

            auto matrix = load_csv(path);
			std::remove(path.c_str());
            Assert::AreEqual("2", std::string(matrix(1, 1)).c_str());
        }

    
	};

    TEST_CLASS(PreprocessingTests)
    {
    public:

        TEST_METHOD(NumericalColumn_Normalization_Test)
        {
            // 1. Arrange: 10, 20, 30 (mean =  20)
            StringMatrix data(4, 1);
            data(0, 0) = "Age";
            data(1, 0) = "10";
            data(2, 0) = "20";
            data(3, 0) = "30";

			NumericalColumn col(0, "Age");
            // 2. Act
            col.fit(data);
            double result = col.transform("30"); // (30-20) / std_dev

            // 3. Assert
            // StdDev dla 10, 20, 30 wynosi ok. 8.16497
            Assert::IsTrue(result > 0, L"Normalized value for 30 should be positive");
			Assert::AreEqual(1.22474424, result, 0.001);
            Assert::AreEqual(0.0, col.transform("20"), 0.001, L"Mean should be transformed to 0");
        }

        TEST_METHOD(CategoricalColumn_Sort_And_Index_Test)
        {
            // 1. Arrange
            StringMatrix data(5, 1);
            data(0, 0) = "City";
            data(1, 0) = "Warszawa";
            data(2, 0) = "Krakow";
            data(3, 0) = "Warszawa";
            data(4, 0) = "Gdansk";

			CategoricalColumn col(0, "City");

            // 2. Act
            col.fit(data);

            // 3. Assert
            Assert::AreEqual(0.0, col.transform("Gdansk"), L"Gdansk should be index 0");
            Assert::AreEqual(1.0, col.transform("Krakow"), L"Krakow should be index 1");
            Assert::AreEqual(2.0, col.transform("Warszawa"), L"Warszawa should be index 2");
            Assert::AreEqual(-1.0, col.transform("Bialystok"), L"Unknown category should return -1");
        }

        TEST_METHOD(DataPreprocessor_Full_Flow_Test)
        {
            // 1. Arrange
            StringMatrix data(3, 3);
            data(0, 0) = "Age"; data(0, 1) = "City"; data(0, 2) = "Salary";
            data(1, 0) = "20";  data(1, 1) = "A";    data(1, 2) = "1000";
            data(2, 0) = "40";  data(2, 1) = "B";    data(2, 2) = "2000";

            DataPreprocessor prep;
            prep.initialize_from_data(data);

			auto& columns = prep.get_columns();
			Assert::AreEqual(size_t(3), columns.size(), L"Should initialize 3 columns");
			columns[0]->include_column = false; // Exclude Age from input features
			columns[2]->is_target_column = true; // Salary as target
            // 2. Act
            prep.fit(data);
            Dataset ds = prep.transform(data);

            // 3. Assert
            Assert::AreEqual(size_t(2), ds.input_data.get_rows_nb());
            Assert::AreEqual(size_t(1), ds.input_data.get_columns_nb());

			Assert::AreEqual(size_t(2), ds.output_data.get_rows_nb());
			Assert::AreEqual(size_t(1), ds.output_data.get_columns_nb());

			Assert::AreEqual(0.0, ds.input_data(0, 0), 0.001, L"City A should be transformed to 0");


        }

        TEST_METHOD(NumericalColumn_Handle_Zero_StdDev_Test)
        {
            StringMatrix data(3, 1);
            data(0, 0) = "Const";
            data(1, 0) = "10";
            data(2, 0) = "10";

			NumericalColumn col(0, "Const");
            col.fit(data);

			// Theoretically, the std_dev should be zero here, but we should handle this in the transform function to avoid division by zero.
            double result = col.transform("10");
            Assert::AreEqual(0.0, result, 0.001, L"Should handle zero variance");
        }

        TEST_METHOD(Handle_Edge_Conditions) {
			// mixed data with numerical column having zero variance, invalid numeric data, and categorical column with new/unseen categories. Also test excluding columns.
            StringMatrix data(5, 3);
            data(0, 0) = "Const"; data(0, 1) = "Mixed"; data(0, 2) = "Cat";

            data(1, 0) = "10.0";  data(1, 1) = "100";   data(1, 2) = "A";
            data(2, 0) = "10.0";  data(2, 1) = "ERROR"; data(2, 2) = "B"; 
            data(3, 0) = "10.0";  data(3, 1) = "200";   data(3, 2) = ""; 
            data(4, 0) = "10.0";  data(4, 1) = "300";   data(4, 2) = "A";

            DataPreprocessor prep;
            prep.initialize_from_data(data);

            
            //for (auto& col : prep.get_columns()) col->include_column = true;

            prep.fit(data);

            Dataset ds = prep.transform(data);

            Assert::AreEqual(0.0, ds.input_data(0, 0), 0.001, L"Zero variance column should result in 0.0");

            double val_mixed = ds.input_data(1, 1);
            Assert::IsFalse(std::isnan(val_mixed), L"Invalid numeric data should not result in NaN");

            auto& cat_col = prep.get_columns()[2];
            double unseen_result = cat_col->transform("Z");
            Assert::AreEqual(-1.0, unseen_result, L"Unseen category should return -1.0");

            prep.get_columns()[1]->include_column = false;
            Dataset ds_filtered = prep.transform(data);

            Assert::AreEqual(size_t(2), ds_filtered.input_data.get_columns_nb(), L"Excluded column should not be in the final matrix");
        }

        TEST_METHOD(Should_Save_Load_Transform_Datapreprocessor) {
            StringMatrix data(4, 2);
            data(0, 0) = "Const"; data(0, 1) = "Mixed";

            data(1, 0) = "10.0";  data(1, 1) = "A";
            data(2, 0) = "20.0";  data(2, 1) = "B";

            DataPreprocessor prep;
            prep.initialize_from_data(data);
            prep.fit(data);
            auto& initial_columns = prep.get_columns();
            prep.save("datapreprocessor_test");
            DataPreprocessor loaded_prep;
            loaded_prep.load("datapreprocessor_test");
            auto& loaded_columns = loaded_prep.get_columns();
            //auto& first_col = dynamic_cast<NumericalColumn*>(&loaded_columns[0]);
        }
    };

    TEST_CLASS(DenseLayerTest) {
    public:

        TEST_METHOD(Dense_Feedforward_Result_Test) {
            Dense layer(0, 2, 1);
            
            layer.get_weights()(0, 0) = 0.5;
            layer.get_weights()(1, 0) = 0.1;
            layer.get_bias()(0, 0) = 0.2;

            Matrix inputs(1, 2);
            inputs(0, 0) = 1.0;
            inputs(0, 1) = 2.0;

            Matrix output = layer.feedforward(inputs);

            // (1.0 * 0.5) + (2.0 * 0.1) + 0.2 = 0.9
            Assert::AreEqual(0.9, output(0, 0), 0.0001, L"Forward pass result is incorrect");
        }

        TEST_METHOD(Dense_Backpropagate_InputGradients_Test) {
            Dense layer(0, 2, 1);
            layer.get_weights()(0, 0) = 0.5;
            layer.get_weights()(1, 0) = 0.1;

            Matrix inputs(1, 2); 

            Matrix grad_next(1, 1);
            grad_next(0, 0) = 1.0;

            //  dL/dX = grad_next * W^T
            Matrix grad_prev = layer.backpropagate(inputs, grad_next);

            Assert::AreEqual(0.5, grad_prev(0, 0), 0.0001, L"Input gradient [0] is incorrect");
            Assert::AreEqual(0.1, grad_prev(0, 1), 0.0001, L"Input gradient [1] is incorrect");
        }

        TEST_METHOD(Dense_Backpropagate_WeightGradients_Accumulation_Test) {
            Dense layer(0, 2, 1);
            Matrix inputs(1, 2);
            inputs(0, 0) = 10.0;
            inputs(0, 1) = 20.0;

            Matrix grad_next(1, 1);
            grad_next(0, 0) = 1.0;

            layer.zero_gradients();

            // dW = inputs^T * grad_next
            layer.backpropagate(inputs, grad_next);

            // dW should be equal [10.0, 20.0]^T
            Assert::AreEqual(10.0, layer.get_accumulated_gradients()(0, 0), 0.0001);
            Assert::AreEqual(20.0, layer.get_accumulated_gradients()(1, 0), 0.0001);
        }
        TEST_METHOD(Dense_Parameters_Update_Test) {
            Dense layer(0, 2, 1);
            layer.get_weights()(0, 0) = 0.5;
            layer.get_weights()(1, 0) = 0.1;
            layer.get_bias()(0, 0) = 0.2;
            Matrix inputs(1, 2);
            inputs(0, 0) = 10.0;
            inputs(0, 1) = 20.0;
            Matrix grad_next(1, 1);
            grad_next(0, 0) = 1.0;
            layer.zero_gradients();
            layer.backpropagate(inputs, grad_next);
            double learning_rate = 0.01;
            size_t batch_size = 1;
            layer.update_params(learning_rate, batch_size);
            // New weights should be W - lr * dW
            Assert::AreEqual(0.5 - learning_rate * 10.0, layer.get_weights()(0, 0), 0.0001);
            Assert::AreEqual(0.1 - learning_rate * 20.0, layer.get_weights()(1, 0), 0.0001);
			Assert::AreEqual(0.2 - learning_rate * grad_next(0, 0), layer.get_bias()(0, 0), 0.0001);
        }
    };

    TEST_CLASS(EmbeddingLayerTests){
        TEST_METHOD(TestInitializationAndDimensions)
        {
            std::vector<EmbeddingConfig> configs = {
                {1, 5, 3}
            };

            TabularEmbeddingLayer layer(1, 2, configs);

            // Output should be: 1 (numeric) + 3 (vectors) = 4
            Assert::AreEqual(static_cast<size_t>(2), layer.get_input_nb(), L"Bad output size");
            Assert::AreEqual(static_cast<size_t>(4), layer.get_output_nb(), L"Bad output size");
        }

        TEST_METHOD(TestFeedforwardPassThrough)
        {
            std::vector<EmbeddingConfig> configs = { {1, 5, 3} };
            TabularEmbeddingLayer layer(1, 2, configs);
            layer.initialize();

            Matrix inputs(2, 2);
            inputs(0, 0) = 42.5; 
            inputs(0, 1) = 2.0;  // Category
            inputs(1, 0) = 15.0; 
            inputs(1, 1) = 0.0;  // Category

            Matrix output = layer.feedforward(inputs);

            Assert::AreEqual(static_cast<size_t>(2), output.get_rows_nb(), L"Bad output row size");
            Assert::AreEqual(static_cast<size_t>(4), output.get_columns_nb(), L"Bad output row size");

            Assert::AreEqual(42.5, output(0, 0), 0.0001, L"Numeric column got changed");
            Assert::AreEqual(15.0, output(1, 0), 0.0001, L"Numeric column got changed");
        }

        TEST_METHOD(TestBackpropagateGradients)
        {
            std::vector<EmbeddingConfig> configs = { {1, 5, 3} };
            TabularEmbeddingLayer layer(1, 2, configs);
            layer.initialize();

            Matrix inputs(1, 2);
            inputs(0, 0) = 10.0; 
            inputs(0, 1) = 1.0;  // Category

            Matrix incoming_gradients(1, 4);
            incoming_gradients(0, 0) = -0.5; 
            incoming_gradients(0, 1) = 0.2;  // Errors for embeddings...
            incoming_gradients(0, 2) = 0.3;  //...
            incoming_gradients(0, 3) = 0.4;  //...

            Matrix input_gradients = layer.backpropagate(inputs, incoming_gradients);

            Assert::AreEqual(0.0, input_gradients(0, 0), 0.0001, L"Embedding gradient should be 0");
            Assert::AreEqual(0.0, input_gradients(0, 1), 0.0001, L"Embedding gradient should be 0");
        }
        TEST_METHOD(TestSerialization_SaveAndLoad)
        {
            // 1. Definiujemy konfiguracjê dla dwóch kolumn kategorycznych
            std::vector<EmbeddingConfig> configs = {
                {0, 5, 3}, // Kolumna 0: 5 unikalnych wartoœci, wymiar = 3
                {2, 10, 4} // Kolumna 2: 10 unikalnych wartoœci, wymiar = 4
            };

            // 2. Tworzymy ORYGINALN¥ warstwê
            // Wejœcia: 3 kolumny, Wyjœcia: 3 + (3-1) + (4-1) = 8
            TabularEmbeddingLayer original_layer(1, 3, configs);
            original_layer.initialize(); // Nape³nia losowymi wagami

            // Tworzymy dane testowe
            Matrix test_inputs(2, 3);
            test_inputs(0, 0) = 2.0;
            test_inputs(0, 1) = 0.5;
            test_inputs(0, 2) = 7.0;

            test_inputs(1, 0) = 4.0;
            test_inputs(1, 1) = 0.9;
            test_inputs(1, 2) = 1.0;

            // Przepuszczamy sygna³ i zapisujemy wynik
            Matrix original_output = original_layer.feedforward(test_inputs);

            // 3. Zapisujemy warstwê do pliku
            std::string test_filename = "test_embedding_layer.bin";
            {
                std::ofstream out(test_filename, std::ios::binary);
                Assert::IsTrue(out.is_open(), L"Nie udalo sie utworzyc pliku testowego!");
                original_layer.save(out);
            }

            // 4. Tworzymy NOW¥, pust¹ warstwê (szkielet z LayerMakera)
            TabularEmbeddingLayer loaded_layer(1, 3, 8);

            // 5. £adujemy wagi z pliku
            {
                std::ifstream in(test_filename, std::ios::binary);
                Assert::IsTrue(in.is_open(), L"Nie udalo sie otworzyc pliku testowego do odczytu!");
                loaded_layer.load(in);
            }

            // 6. Przepuszczamy TE SAME dane
            Matrix loaded_output = loaded_layer.feedforward(test_inputs);

            // 7. Weryfikacja Asercjami (Assert)
            Assert::AreEqual(original_output.get_rows_nb(), loaded_output.get_rows_nb(), L"Niezgodna liczba wierszy (Batch Size)");
            Assert::AreEqual(original_output.get_columns_nb(), loaded_output.get_columns_nb(), L"Niezgodna liczba kolumn wyjsciowych");

            // Sprawdzamy zawartoœæ macierzy (z tolerancj¹ na b³êdy zmiennoprzecinkowe)
            for (size_t r = 0; r < original_output.get_rows_nb(); ++r) {
                for (size_t c = 0; c < original_output.get_columns_nb(); ++c) {

                    double diff = std::abs(original_output(r, c) - loaded_output(r, c));

                    // Wiadomoœæ b³êdu w razie pomy³ki
                    std::wstring error_msg = L"Blad na pozycji (" + std::to_wstring(r) + L", " + std::to_wstring(c) + L")";

                    // Tolerancja 1e-9 (bardzo ostra)
                    Assert::IsTrue(diff < 1e-9, error_msg.c_str());
                }
            }

            // Opcjonalnie: Czyszczenie (usuniêcie pliku testowego z dysku po udanym teœcie)
            std::remove(test_filename.c_str());
        }
    };

    TEST_CLASS(NeuralNetworkTest) {
    public:
        TEST_METHOD(Predict_DenseLayers_Test) {
            NeuralNetwork nn;

            auto l1 = std::make_unique<Dense>(0, 2, 2);
            auto l2 = std::make_unique<Dense>(1, 2, 1);

            l1->get_weights()(0, 0) = 0.1; l1->get_weights()(0, 1) = 0.2;
            l1->get_weights()(1, 0) = 0.3; l1->get_weights()(1, 1) = 0.4;
            l1->get_bias()(0, 0) = 0.5;    l1->get_bias()(0, 1) = 0.5;

            l2->get_weights()(0, 0) = 0.5;
            l2->get_weights()(1, 0) = 1.0;
            l2->get_bias()(0, 0) = 0.1;

            nn.add_layer(std::move(l1));
            nn.add_layer(std::move(l2));

            Matrix inputs(1, 2);
            inputs(0, 0) = 1.0;
            inputs(0, 1) = 0.5;

            Matrix result = nn.predict(inputs);

            Assert::AreEqual(1.375, result(0, 0), 0.0001, L"NeuralNetwork predict result mismatch!");

            Assert::AreEqual((size_t)1, result.get_rows_nb());
            Assert::AreEqual((size_t)1, result.get_columns_nb());
        }

        TEST_METHOD(Predict_Without_Layers_Should_Throw) {
            NeuralNetwork nn;
            Matrix inputs(1, 2);
            inputs(0, 0) = 1.0;
            inputs(0, 1) = 0.5;
            auto func = [&nn, &inputs]() { nn.predict(inputs); };
            Assert::ExpectException<std::runtime_error>(func, L"Predicting without layers should throw an exception");
		}

        TEST_METHOD(NeuralNetwork_Loss_and_Test_Test) {
            NeuralNetwork nn;

            nn.set_loss(std::make_unique<MSE>());

            auto layer = std::make_unique<Dense>(0, 1, 1);
			layer->get_weights()(0, 0) = 2.0;
			layer->get_bias()(0, 0) = 0.0;

            nn.add_layer(std::move(layer));
            // Data: input [1.0], target [3.0]
            // Prediction: 1.0 * 2.0 = 2.0
            // Error (MSE): (2.0 - 3.0)^2 = 1.0
            Matrix in(1, 1); in(0, 0) = 1.0;
            Matrix target(1, 1); target(0, 0) = 3.0;

            double result = nn.test(in, target);

            Assert::AreEqual(1.0, result, 0.0001, L"Loss calculation via Loss class failed!");
        }
    };

    TEST_CLASS(ModelIOTests)
    {
    public:
        TEST_METHOD(Preprocessor_SaveAndLoad_ShouldPreserveData)
        {
            const std::string test_file = "test_preprocessor_config.json";

            DataPreprocessor p1;
			StringMatrix dummy_data(3, 2);

			dummy_data(0, 0) = "Age"; dummy_data(0, 1) = "City";
			dummy_data(1, 0) = "20";  dummy_data(1, 1) = "A";
			dummy_data(2, 0) = "40";  dummy_data(2, 1) = "B";

            p1.initialize_from_data(dummy_data);
            p1.fit(dummy_data);
            p1.save(test_file);

            DataPreprocessor p2;
            p2.load(test_file);

            const auto& cols1 = p1.get_columns();
            const auto& cols2 = p2.get_columns();

            Assert::AreEqual(cols1.size(), cols2.size(), L"Number of columns does not match!");
            Assert::AreEqual(cols1[0]->name, cols2[0]->name, L"Column names have changed!");

            auto* num_col = dynamic_cast<NumericalColumn*>(cols2[0].get());
            Assert::IsNotNull(num_col, L"The first column should be Numerical!");
            Assert::AreEqual(30.0, num_col->get_mean(), 1e-6, L"Incorrect mean after loading JSON!");

            auto* cat_col = dynamic_cast<CategoricalColumn*>(cols2[1].get());
            Assert::IsNotNull(cat_col, L"The second column should be Categorical!");
            Assert::AreEqual(static_cast<size_t>(2), cat_col->get_categories().size(), L"Incorrect number of categories!"); 
            
            auto load_invalid_file = [] {
                DataPreprocessor p3;
                p3.load("not_existent.json");
                };
            Assert::ExpectException<std::runtime_error>(load_invalid_file, L"No exception thrown for missing file!");

            std::remove(test_file.c_str());
        }

        TEST_METHOD(NeuralNetwork_SaveAndLoad_ShouldPreserveWeights)
        {
            const std::string test_file = "test_network_weights.bin";

            NeuralNetwork nn1;
            nn1.add_layer(LayerMaker::create_by_name("Dense", 0, { 0, 2, 4, 0.0f }));
            nn1.add_layer(LayerMaker::create_by_name("ReLU", 1, { 0, 4, 4, 0.0f }));
            nn1.add_layer(LayerMaker::create_by_name("Dense", 2, { 0, 4, 1, 0.0f }));
            nn1.add_layer(LayerMaker::create_by_name("Sigmoid", 3, { 0, 1, 1, 0.0f }));

            Matrix dummy_input(1, 2);
            dummy_input(0, 0) = 0.5;
            dummy_input(0, 1) = -0.5;

            Matrix output1 = nn1.predict(dummy_input);
            nn1.save(test_file);

            NeuralNetwork nn2;
            nn2.load(test_file);
            Matrix output2 = nn2.predict(dummy_input);

            Assert::AreEqual(output1.get_rows_nb(), output2.get_rows_nb(), L"Bad output rows!");
            Assert::AreEqual(output1.get_columns_nb(), output2.get_columns_nb(), L"Bad output columns!");

            // If binary dumping works perfectly, the numbers must match 100% up to the 9th decimal place
            Assert::AreEqual(output1(0, 0), output2(0, 0), 1e-9, L"Difference in results! Weights did not load precisely.");

            auto load_invalid_model = [] {
                NeuralNetwork nn3;
                nn3.load("non_existent_file.bin");
                };
            Assert::ExpectException<std::runtime_error>(load_invalid_model, L"No exception thrown for missing binary file!");

            std::remove(test_file.c_str());
        }
    };
}
