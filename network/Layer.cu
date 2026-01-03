//
// Created by martin on 26.12.25.
//

#include "Layer.h"

#include <stdexcept>

#include "../utils/NeuralNetworkUtils.h"

Vector Layer::activate(const Vector &X) {
    this->X = X;
    this->Z = this->W * this->X + this->B;
    this->A = NeuralNetworkUtils::applyActivation(this->activationFunction, this->Z);
    return this->A;
}

Vector Layer::calcDelta(const Vector& classes, ErrorFunction errorFunction) {
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
    Vector error = layer.getW().transpose() * layer.getD();
    this->D = error * NeuralNetworkUtils::applyActivationDerivative(this->activationFunction,nullptr,this->Z);
    return this->D;
}

Matrix Layer::calcGradients() {
    this->G = this->D ^ this->X;
    return this->G;
}

void Layer::adaptWeights(const Matrix &gradients, float learningRate) {
    this->W -= gradients^learningRate;
}

void Layer::adaptBiases(const Vector &deltas, float learningRate) {
    this->B = this->B - deltas *= learningRate;
}
