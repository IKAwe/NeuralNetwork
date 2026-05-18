#pragma once
#include <vector>
#include <string_view>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <omp.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif


/**
 * @brief For usage especially when we dont know the number of rows. First - add al the cells with add_cell, then set the number of columns with set_columns_nb, 
 and finally call finalize to calculate the number of rows based on the number of cells and columns.
 */
class StringMatrix {
	std::vector<char> buffer;
	std::vector<std::string_view> cells;

	size_t rows = 0;
	size_t columns = 0;
public:
	StringMatrix(size_t r, size_t c) : rows(r), columns(c) {
		cells.resize(r * c);
		buffer.reserve(r * c * 10); // Assuming average cell length of 10 characters, adjust as needed
	};
	StringMatrix() = default;
	/**
 * @brief Add a string_view cell to cells vector. The string_view should point to a valid string in the buffer vector.
 * @param cell string_view to be added to cells vector.
 */
	void add_cell(std::string_view cell) {
		cells.push_back(cell);
	};
	/**
 * @brief Define the number of columns in the matrix.
 * @param c The number of columns in the matrix.
 */
	void set_columns_nb(size_t c) {
		columns = c;
	};
	/**
 * @brief Reserve space in the cells vector for a given number of rows and columns. This is an optimization to avoid multiple reallocations as cells are added to the matrix.
 * @param r Rows to reserve space for.
 * @param c Columns to reserve space for.
 */
	void reserve_cells(size_t number_of_cells) {
		cells.reserve(number_of_cells);
	};
	/**
 * @brief
 * @return A reference to the buffer vector.
 */
	std::vector<char>& get_buffer() { return buffer; };

	/**
	 * @brief Finalize the matrix by calculating the number of rows based on the number of cells and columns. This should be called after all cells have been added and the number of columns has been set.
	 */
	void finalize() { rows = cells.size() / columns; };


	//Getters
	size_t get_rows_nb() const {
		return rows;
	};
	size_t get_columns_nb() const {
		return columns;
	};
	//For writing to the matrix
	std::string_view& operator()(size_t r, size_t c) {
		return cells[r * columns + c];
	}
	//For reading from the matrix
	std::string_view operator()(size_t r, size_t c) const {
		return cells[r * columns + c];
	}
};

class Matrix{
	std::vector<double> data;
	size_t rows_nb = 0;
	size_t col_nb = 0;
public:
	//Matrix() = default;
	Matrix(size_t r, size_t c) : rows_nb(r), col_nb(c), data(r* c, 0.0) {};
	//for saving
	double& operator()(size_t r, size_t c) {
		if (r >= rows_nb || c >= col_nb) {
			throw std::out_of_range("Matrix index out of range");
		}
		return data[r * col_nb + c];
	}
	//for reading
	double operator()(size_t r, size_t c) const {
		if (r >= rows_nb || c >= col_nb) {
			throw std::out_of_range("Matrix index out of range");
		}
		return data[r * col_nb + c];
	}
	//Used for X*W
	Matrix operator*(const Matrix& other) const {
#ifndef USE_CUDA 
		if (col_nb != other.rows_nb) {
			throw std::runtime_error("Matrix dimensions do not match for multiplication");
		}
		const double* A = data.data();
		const double* B = other.data.data();
		double* C = result.data.data();

		int r_nb = (int)rows_nb;
		int c_nb = (int)col_nb;
		int other_c_nb = (int)other.col_nb;

		Matrix result(rows_nb, other.col_nb);
		const double* A = data.data();
		const double* B = other.data.data();
		double* C = result.data.data();

		int r_nb = (int)rows_nb;
		int c_nb = (int)col_nb;
		int other_c_nb = (int)other.col_nb;

#pragma omp parallel for
		for (int i = 0; i < r_nb; ++i) {
			for (int k = 0; k < c_nb; ++k) {
				double a_ik = A[i * c_nb + k];
				for (int j = 0; j < other_c_nb; ++j) {
					C[i * other_c_nb + j] += a_ik * B[k * other_c_nb + j];
				}
			}
		}
		return result;
#else
		Matrix result(rows_nb, other.col_nb);

		// 1. Alokacja pamięci na GPU
		double *d_A, *d_B, *d_C;
		size_t sizeA = rows_nb * col_nb * sizeof(double);
		size_t sizeB = other.rows_nb * other.col_nb * sizeof(double);
		size_t sizeC = rows_nb * other.col_nb * sizeof(double);

		cudaMalloc(&d_A, sizeA);
		cudaMalloc(&d_B, sizeB);
		cudaMalloc(&d_C, sizeC);

		// 2. Kopiowanie danych CPU → GPU
		cudaMemcpy(d_A, data.data(), sizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, other.get_data().data(), sizeB, cudaMemcpyHostToDevice);

		// 3. Wywołanie cuBLAS GEMM
		static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
	 
		const double alpha = 1.0;
		const double beta  = 0.0;

		// Uwaga: cuBLAS używa column-major, więc trzeba transponować logikę
		cublasDgemm(
			handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			other.col_nb, rows_nb, col_nb,
			&alpha,
			d_B, other.col_nb,
			d_A, col_nb,
			&beta,
			d_C, other.col_nb
		);

		// 4. Kopiowanie wyników GPU - CPU
		cudaMemcpy(result.get_data_mutable().data(), d_C, sizeC, cudaMemcpyDeviceToHost);

		// 5. Sprzątanie
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);

		return result;
#endif
	}

	Matrix operator*(const double scalar) const {
		Matrix result(rows_nb, col_nb);
		int n = (int)data.size();
		const double* src = data.data();
		double* dst = result.data.data();

#pragma omp parallel for simd
		for (int i = 0; i < n; ++i) {
			dst[i] = src[i] * scalar;
		}
		return result;
	}
	Matrix operator-(const Matrix& other) const {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Matrix dimensions do not match for subtraction");
		}
		Matrix result(rows_nb, col_nb);
		int n = (int)data.size();
		const double* A = data.data();
		const double* B = other.data.data();
		double* C = result.data.data();

#pragma omp parallel for simd
		for (int i = 0; i < n; ++i) {
			C[i] = A[i] - B[i];
		}
		return result;
	}
	Matrix operator+(const Matrix& other) const {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Matrix dimensions do not match for addition");
		}
		Matrix result(rows_nb, col_nb);
		int n = (int)data.size();
		const double* A = data.data();
		const double* B = other.data.data();
		double* C = result.data.data();

#pragma omp parallel for simd
		for (int i = 0; i < n; ++i) {
			C[i] = A[i] + B[i];
		}
	}
	Matrix transpose() const {
		Matrix result(col_nb, rows_nb);
		for (size_t r = 0; r < rows_nb; ++r) {
			for (size_t c = 0; c < col_nb; ++c) {
				result(c, r) = (*this)(r, c);
			}
		}
		return result;
	}
	Matrix hadamard(const Matrix& other) const {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Dimensions must match for Hadamard product");
		}
		Matrix result(rows_nb, col_nb);
		int n = (int)data.size();
		const double* A = data.data();
		const double* B = other.data.data();
		double* C = result.data.data();

#pragma omp parallel for simd
		for (int i = 0; i < n; ++i) {
			C[i] = A[i] * B[i];
		}
		return result;
	}

	Matrix& operator+=(const Matrix& other) {
		if (rows_nb != other.rows_nb || col_nb != other.col_nb) {
			throw std::runtime_error("Dimensions do not match for addition");
		}
		int n = (int)data.size();
		double* A = data.data();
		const double* B = other.data.data();

#pragma omp parallel for simd
		for (int i = 0; i < n; ++i) {
			A[i] += B[i];
		}
		return *this;
	}
	void zero() {
		std::fill(data.begin(), data.end(), 0.0);
	}
	void resize(size_t new_rows, size_t new_cols) {
		rows_nb = new_rows;
		col_nb = new_cols;
		data.resize(new_rows * new_cols, 0.0);
	}
	void print() const {
		for (size_t r = 0; r < rows_nb; ++r) {
			std::cout << "[";
			for (size_t c = 0; c < col_nb; ++c) {
				std::cout << std::setw(6) << std::right << (*this)(r, c) << " ";
			}
			std::cout << "]" << std::endl;
		}
	}
	 const std::vector<double>& get_data() const {
		return data;
	}
	 std::vector<double>& get_data_mutable() {
		 return data;
	 }

	 void overwrite_with_rows(const Matrix& source, size_t start_row, size_t end_row) {

		 size_t new_rows = end_row - start_row;

		 this->rows_nb = new_rows;
		 this->col_nb = source.get_columns_nb();
		 this->data.resize(new_rows * this->col_nb);

		 std::copy(source.data.begin() + (start_row * source.get_columns_nb()),
					source.data.begin() + (end_row * source.get_columns_nb()),
					this->data.begin());

	 }
	 void save(std::ostream& out) const {
		 out.write(reinterpret_cast<const char*>(&rows_nb), sizeof(rows_nb));
		 out.write(reinterpret_cast<const char*>(&col_nb), sizeof(col_nb));
		 out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
	 }

	 void load(std::istream& in) {
		 in.read(reinterpret_cast<char*>(&rows_nb), sizeof(rows_nb));
		 in.read(reinterpret_cast<char*>(&col_nb), sizeof(col_nb));
		 data.resize(rows_nb * col_nb);
		 in.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(double));
	 }

	size_t get_rows_nb() const { return rows_nb; }
	size_t get_columns_nb() const { return col_nb; }
};