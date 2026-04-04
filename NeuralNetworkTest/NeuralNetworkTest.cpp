#include "pch.h"
#include <fstream>
#include "CppUnitTest.h"
#include "csv_loader.h"
#include "data_structures.h"
#include "data_preprocessor.h"

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

        TEST_METHOD(Should_Handle_Windows_Endings) {
            std::string path = "win.csv";
            std::ofstream f(path, std::ios::binary);
            f << "col1,col2\r\n1,2\r\n";
            f.close();

            auto matrix = load_csv(path);

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

            NumericalColumn col;
            col.index = 0;

            // 2. Act
            col.fit(data);
            double result = col.transform("30"); // (30-20) / std_dev

            // 3. Assert
            Assert::AreEqual(20.0, col.mean, 0.001, L"Mean should be 20");
			Assert::AreEqual(8.16497, col.std_dev, 0.001, L"StdDev should be approximately 8.16497");
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

            CategoricalColumn col;
            col.index = 0;

            // 2. Act
            col.fit(data);

            // 3. Assert
            Assert::AreEqual(size_t(3), col.categories.size(), L"Should find 3 unique categories");
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

            NumericalColumn col;
            col.index = 0;
            col.fit(data);

			// Theoretically, the std_dev should be zero here, but we should handle this in the transform function to avoid division by zero.
            double result = col.transform("10");
            Assert::AreEqual(0.0, result, 0.001, L"Should handle zero variance");
        }

        TEST_METHOD(Handle_Edge_Conditions) {
			// Arrange: mixed data with numerical column having zero variance, invalid numeric data, and categorical column with new/unseen categories. Also test excluding columns.
            StringMatrix data(5, 3);
            data(0, 0) = "Const"; data(0, 1) = "Mixed"; data(0, 2) = "Cat";

            data(1, 0) = "10.0";  data(1, 1) = "100";   data(1, 2) = "A";
            data(2, 0) = "10.0";  data(2, 1) = "ERROR"; data(2, 2) = "B"; 
            data(3, 0) = "10.0";  data(3, 1) = "200";   data(3, 2) = ""; 
            data(4, 0) = "10.0";  data(4, 1) = "300";   data(4, 2) = "A";

            DataPreprocessor prep;
            prep.initialize_from_data(data);

            
            //for (auto& col : prep.get_columns()) col->include_column = true;

            // --- 2. Test Fit  ---
            prep.fit(data);

            // --- 3. Test Transform  ---
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
    };
}
