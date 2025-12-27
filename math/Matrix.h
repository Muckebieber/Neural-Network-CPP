//
// Created by martin on 26.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_MATRIX_H
#define NEURONAL_NETWORK_CPP_MATRIX_H
#include <vector>

#include "Vector.h"


class Matrix {
    public:
        Matrix(size_t rows, size_t columns) : rows(rows), columns(columns), matrix(rows, std::vector<float>(columns,0.0)) {};
        static void zeroes();
        static void ones();
        static void random();
        size_t getRows() const;
        size_t getCols() const;
        Vector operator*(const Vector& other) const;
        std::vector<float> operator[](size_t row) const;
    private:
        std::vector<std::vector<float>> matrix;
        size_t rows;
        size_t columns;

};


#endif //NEURONAL_NETWORK_CPP_MATRIX_H