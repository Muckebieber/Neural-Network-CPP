//
// Created by martin on 26.12.25.
//

#include "Matrix.h"

#include <stdexcept>

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
