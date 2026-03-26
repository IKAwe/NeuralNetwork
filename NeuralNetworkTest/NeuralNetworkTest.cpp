#include "pch.h"
#include <fstream>
#include "CppUnitTest.h"
#include "csv_loader.h"
#include "data_structures.h"

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
}
