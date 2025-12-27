//
// Created by martin on 26.12.25.
//

#include "Vector.h"

#include <stdexcept>

#include "Matrix.h"

Vector Vector::zeroes(size_t size) {
    Vector v(size);
    for (size_t i = 0; i < size; i++) {
        v[i] = 0;
    }
    return v;
}

Vector Vector::ones(size_t size) {
    Vector v(size);
    for (size_t i = 0; i < size; i++) {
        v[i] = 1;
    }
    return v;
}

Vector Vector::random(size_t size, float min, float max) {
    Vector v(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < size; i++) {
        v[i] = dis(gen);
    }
    return v;
}

size_t Vector::getSize() const {
    return data.size();
}

float& Vector::operator[](size_t index) {
    return this->data[index];
}

const float& Vector::operator[](size_t index) const {
    return this->data[index];
}
Vector Vector::operator+(const Vector& v) const {
    if (this->getSize()==v.getSize()) {
        Vector result(this->getSize());
        for (int i = 0; i < this->getSize(); ++i) {
            result[i] = (*this)[i] + v[i];
        }
        return result;
    }
    throw std::runtime_error("Illegal Argument Exception");
}
Vector Vector::operator-(const Vector& v) const {
    if (this->getSize()==v.getSize()) {
        Vector result(this->getSize());
        for (int i = 0; i < this->getSize(); ++i) {
            result[i] = (*this)[i] - v[i];
        }
        return result;
    }
    throw std::runtime_error("Illegal Argument Exception");
}

Vector Vector::operator*(const Vector& v) const {
    if (this->getSize()==v.getSize()) {
        Vector result(this->getSize());
        for (int i = 0; i < this->getSize(); ++i) {
            result[i] = (*this)[i] * v[i];
        }
        return result;
    }
    throw std::runtime_error("Illegal Argument Exception");
}

Matrix Vector::operator^(const Vector& v) const {
    Matrix result(this->getSize(), v.getSize());
    for (size_t i = 0; i < this->getSize(); ++i) {
        for (size_t j = 0; j < v.getSize(); ++j) {
            result[i][j] = (*this)[i] * v[j];
        }
    }
    return result;
}
