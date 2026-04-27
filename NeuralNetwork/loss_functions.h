#pragma once

#ifndef LOSS_H
#define LOSS_H

#include "data_structures.h"

class Loss {
public:
    virtual ~Loss() = default;

    virtual double calculate(const Matrix& predictions, const Matrix& targets) = 0;
    virtual Matrix calculate_gradient(const Matrix& predictions, const Matrix& targets) = 0;
};

class MSE : public Loss {
public:
    double calculate(const Matrix& predictions, const Matrix& targets) override {
        double sum = 0;
        size_t n = predictions.get_rows_nb() * predictions.get_columns_nb();

        for (size_t r = 0; r < predictions.get_rows_nb(); ++r) {
            for (size_t c = 0; c < predictions.get_columns_nb(); ++c) {
                double diff = predictions(r, c) - targets(r, c);
                sum += diff * diff;
            }
        }
        return sum / static_cast<double>(n);
    }

    Matrix calculate_gradient(const Matrix& predictions, const Matrix& targets) override {
        Matrix grad(predictions.get_rows_nb(), predictions.get_columns_nb());
        size_t n = predictions.get_rows_nb() * predictions.get_columns_nb();
        double factor = 2.0 / static_cast<double>(n);

        for (size_t r = 0; r < grad.get_rows_nb(); ++r) {
            for (size_t c = 0; c < grad.get_columns_nb(); ++c) {
                grad(r, c) = factor * (predictions(r, c) - targets(r, c));
            }
        }
        return grad;
    }
};

#endif