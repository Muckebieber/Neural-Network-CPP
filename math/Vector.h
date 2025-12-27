//
// Created by martin on 26.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_VECTOR_H
#define NEURONAL_NETWORK_CPP_VECTOR_H
#include <vector>
#include <random>

#include "Matrix.h"

class Vector {
    public:
        Vector(size_t size):data(size){}
        static Vector zeroes(size_t size);
        static Vector ones(size_t size);
        static Vector random(size_t size, float min, float max);
        Vector operator+(const Vector& other)const;
        Vector operator*(const Vector& other)const;
        Matrix operator^(const Vector& other)const;
        Vector operator-(const Vector& other)const;
        float& operator[](size_t index);
        const float& operator[](size_t index) const;
        size_t getSize() const;
    private:
        std::vector<float> data;
};


#endif //NEURONAL_NETWORK_CPP_VECTOR_H