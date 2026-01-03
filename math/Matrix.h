//
// Created by martin on 26.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_MATRIX_H
#define NEURONAL_NETWORK_CPP_MATRIX_H
#include <vector>
#include "Vector.h"

class Vector;

class Matrix {
    public:
        Matrix() : rows(0), columns(0) {}
        Matrix(size_t rows, size_t columns) : rows(rows), columns(columns), matrix(rows, std::vector<float>(columns,0.0)) {};
        static Matrix zeroes(size_t rows, size_t cols);
        static Matrix ones(size_t rows, size_t cols);
        static Matrix random(size_t rows, size_t cols, float min, float max);
        size_t getRows() const;
        size_t getCols() const;
        Vector operator*(const Vector& other) const;
        Matrix operator^(float scalar) const;
        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);
        Matrix& operator/=(float scalar);
        std::vector<float>& operator[](size_t row);
        const std::vector<float>& operator[](size_t row) const;
        Matrix transpose() const;
    private:
        std::vector<std::vector<float>> matrix;
        size_t rows;
        size_t columns;

};


#endif //NEURONAL_NETWORK_CPP_MATRIX_H