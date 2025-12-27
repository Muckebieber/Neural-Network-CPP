//
// Created by martin on 26.12.25.
//
#ifndef NEURONAL_NETWORK_CPP_NETWORK_H
#define NEURONAL_NETWORK_CPP_NETWORK_H
#include "../math/Vector.h"
#include "../math/Matrix.h"
#include "../enums/ActivationFunction.h"

class Network {
    public:
        Network();
        Vector predict(const Vector& X);
        void train(const Vector& features, const Vector& classes, int iterations, int batchSize);
    private:
        void adaptParameters(const Matrix& gradients, const Matrix& deltas);

};


#endif //NEURONAL_NETWORK_CPP_NETWORK_H