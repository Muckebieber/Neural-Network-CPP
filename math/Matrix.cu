//
// Created by martin on 26.12.25.
//

#include "Matrix.h"

#include <stdexcept>

Matrix Matrix::zeroes(size_t rows , size_t cols) {
    Matrix result(rows,cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = 0;
        }
    }
    return result;
}

Matrix Matrix::ones(size_t rows , size_t cols) {
    Matrix result(rows,cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = 1;
        }
    }
    return result;
}

Matrix Matrix::random(size_t rows , size_t cols, float min , float max) {
    Matrix result(rows,cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = dis(gen);
        }
    }
    return result;
}

size_t Matrix::getRows() const {
    return this->rows;
}

size_t Matrix::getCols() const {
    return this->columns;
}

std::vector<float>& Matrix::operator[](size_t row) {
    return this->matrix[row];
}

const std::vector<float>& Matrix::operator[](size_t row) const {
    return this->matrix[row];
}

Vector Matrix::operator*(const Vector &other) const {
    if (this->getCols() == other.getSize()) {
        Vector result(this->getRows());
        for (size_t i = 0; i < this->getRows(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < this->getCols(); j++) {
                sum += (*this)[i][j] * other[j];
            }
            result[i] = sum;
        }
        return result;
    }
    throw std::runtime_error("Illegal Argument Exception");
}

Matrix& Matrix::operator+=(const Matrix &other) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            (*this)[i][j] += other[i][j];
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix &other) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            (*this)[i][j] -= other[i][j];
        }
    }
    return *this;
}

Matrix &Matrix::operator/=(float scalar) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            (*this)[i][j] /= scalar;
        }
    }
    return *this;
}

Matrix Matrix::operator^(float scalar) const {
    Matrix result(this->getRows(),this->getCols());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            result[i][j] *= scalar;
        }
    }
    return *this;
}
