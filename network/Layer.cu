//
// Created by martin on 26.12.25.
//

#include "Layer.h"

#include <stdexcept>

#include "../utils/NeuralNetworkUtils.h"

Vector Layer::activate(Vector &X) {
    this->X = X;
    this->Z = W * X;
    this->Z =this->Z + this->B;
    this->A = NeuralNetworkUtils::applyActivation(this->activationFunction,X);
    return this->A;
}

Vector Layer::calcDelta(Vector& classes, ErrorFunction errorFunction) {
    switch (errorFunction) {
        case MSE:
            this->D = NeuralNetworkUtils::MSEDerivative(classes,this->A) * NeuralNetworkUtils::applyActivationDerivative(this->activationFunction,&classes,this->Z);
            return this->D;
        case CE:
            this->D = NeuralNetworkUtils::softMaxDerivative(classes,this->A);
            return this->D;
    }
    throw new std::runtime_error("calcDelta(): Error");
}

Vector Layer::calcDelta(Layer &layer) {
    Vector error = layer.getW() * layer.getD();
    this->D = error * NeuralNetworkUtils::applyActivationDerivative(this->activationFunction,nullptr,this->Z);
    return this->D;
}

Matrix Layer::calcGradients() {
    this->G = this->D ^ this->Z;
    return this->G;
}
