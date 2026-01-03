//
// Created by martin on 26.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_VECTOR_H
#define NEURONAL_NETWORK_CPP_VECTOR_H
#include <vector>
#include <random>
#include "iostream"
#include "Matrix.h"

class Matrix;

class Vector {
    public:
        Vector() : data() {};
        Vector(size_t size):data(size){}
        static Vector zeroes(size_t size);
        static Vector ones(size_t size);
        static Vector random(size_t size, float min, float max);
        Vector operator+(const Vector& other)const;
        Vector& operator+=(const Vector& other);
        Vector operator*(const Vector& other)const;
        Vector operator*=(float scalar)const;
        Matrix operator^(const Vector& other)const;
        Vector operator-(const Vector& other)const;
        Vector operator/=(float scalar);
        float& operator[](size_t index);
        const float& operator[](size_t index) const;
        size_t getSize() const;
        std::string toString()const;
    private:
        std::vector<float> data;
};


#endif //NEURONAL_NETWORK_CPP_VECTOR_H