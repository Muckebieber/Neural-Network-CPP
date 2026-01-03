//
// Created by martin on 27.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_NEURALNETWORKUTILS_H
#define NEURONAL_NETWORK_CPP_NEURALNETWORKUTILS_H
#include "../math/Vector.h"
#include "../enums/ActivationFunction.h"

class NeuralNetworkUtils {
    public:
        static Vector sigmoid(Vector &X);
        static float sigmoid(float &X);
        static Vector leakyReLU(Vector &X);
        static Vector softMax(Vector &X);
        static float softMax(float& x, Vector& X);
        static float MSE(Vector& X_true, Vector& X);
        static float CE(Vector& X_true, Vector& X);
        static Vector sigmoidDerivative(Vector& X);
        static Vector leakyReLUDerivative(Vector& X);
        static Vector softMaxDerivative(const Vector &X_true, Vector &X);
        static Vector MSEDerivative(const Vector &X_true, Vector &X);
        static Vector applyActivation(ActivationFunction aFunc, Vector &X);
        static Vector applyActivationDerivative(ActivationFunction aFunc, const Vector *X_true, Vector &X);
};


#endif //NEURONAL_NETWORK_CPP_NEURALNETWORKUTILS_H