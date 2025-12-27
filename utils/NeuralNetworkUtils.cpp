//
// Created by martin on 27.12.25.
//

#include "NeuralNetworkUtils.h"

#include <stdexcept>

Vector NeuralNetworkUtils::sigmoid(Vector& X) {
    Vector result(X.getSize());
    for (size_t i = 0; i < X.getSize(); i++) {
        result[i] = 1/(1+exp(-X[i]));
    }
    return result;
}

float NeuralNetworkUtils::sigmoid(float& x) {
    return 1/(1+exp(-x));
}

Vector NeuralNetworkUtils::leakyReLU(Vector& X) {
    Vector result(X.getSize());
    for (size_t i = 0; i < X.getSize(); i++) {
        if (X[i] >= 0.0) {
            result[i] = X[i];
        }
        result[i] = 0.01*X[i];
    }
    return result;
}

Vector NeuralNetworkUtils::softMax(Vector &X) {
    Vector result(X.getSize());
    float sum = 0.0f;
    for (size_t i = 0; i < X.getSize(); i++) {
        sum += exp(X[i]);
    }
    for (size_t i = 0; i < X.getSize(); i++) {
        result[i] = exp(X[i])/sum;
    }
    return result;
}

float NeuralNetworkUtils::softMax(float& x, Vector& X) {
    float sum = 0.0f;
    for (size_t i = 0; i < X.getSize(); i++) {
        sum += exp(X[i]);
    }
    return exp(x)/sum;
}
float NeuralNetworkUtils::MSE(Vector& X_true, Vector& X) {
    float sum = 0.0f;
    for (size_t i = 0; i < X.getSize(); i++) {
        sum += pow(X_true[i]-X[i],2);
    }
    return sum/X.getSize()/2;
}

float NeuralNetworkUtils::CE(Vector& X_true, Vector& X) {
    float sum = 0.0f;
    for (size_t i = 0; i < X.getSize(); i++) {
        sum += X_true[i] * log(X[i]);
    }
    return sum*-1.0/X.getSize();
}

Vector NeuralNetworkUtils::sigmoidDerivative(Vector& X) {
    Vector result(X.getSize());
    for (size_t i = 0; i < X.getSize(); i++) {
        result[i] = sigmoid(X[i]) * (1-sigmoid(X[i]));
    }
    return result;
}

Vector NeuralNetworkUtils::leakyReLUDerivative(Vector& X) {
    Vector result(X.getSize());
    for (size_t i = 0; i < X.getSize(); i++) {
        if (X[i] >= 0.0) {
            result[i] = 1.0;
        } else {
            result[i] = 0.01;
        }
    }
    return result;
}

Vector NeuralNetworkUtils::softMaxDerivative(Vector& X_true, Vector& X) {
    Vector result(X.getSize());
    for (int i = 0; i < X.getSize(); ++i) {
        result[i] = X[i] - X_true[i];
    }
    return result;
}

Vector NeuralNetworkUtils::MSEDerivative(Vector& X_true, Vector &X) {
    return X-X_true;
}

Vector NeuralNetworkUtils::applyActivation(ActivationFunction aFunc, Vector& X) {
    Vector result(X.getSize());
    switch (aFunc) {
        case Sigmoid:
            return sigmoid(X);
        case LeakyReLU:
            return leakyReLU(X);
        case SoftMax:
            return softMax(X);
    }
    throw new std::runtime_error("applyActivation(): Error!");
}
Vector NeuralNetworkUtils::applyActivationDerivative(ActivationFunction aFunc,Vector* X_true=nullptr, Vector& X) {
    Vector result(X.getSize());
    switch (aFunc) {
        case Sigmoid:
            return sigmoidDerivative(X);
        case LeakyReLU:
            return leakyReLUDerivative(X);
        case SoftMax:
            return softMaxDerivative(*X_true,X);
    }
    throw new std::runtime_error("applyActivation(): Error!");
}