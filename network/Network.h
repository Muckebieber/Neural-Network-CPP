//
// Created by martin on 26.12.25.
//
#ifndef NEURONAL_NETWORK_CPP_NETWORK_H
#define NEURONAL_NETWORK_CPP_NETWORK_H
#include "Layer.h"
#include "../math/Vector.h"
#include "../math/Matrix.h"
#include "../enums/ActivationFunction.h"

class Network {
    public:
        Network(size_t inputs, size_t outputs, size_t hiddenNeurons, size_t hiddenLayers, ActivationFunction hiddenActivation, ActivationFunction outputActivaton, ErrorFunction eFunc, float learningRate);
        Vector predict(const Vector& X);
        void train(const std::vector<Vector> &features, const std::vector<Vector> &classes, int iterations, int batchSize);
    private:
        ErrorFunction errorFunction;
        std::vector<Matrix> gradients;
        std::vector<Vector> deltas;
        void backpropagate(const Vector &X, const Vector& O_True);
        void adaptParameters(const std::vector<Matrix> &gradients, const std::vector<Vector> &deltas);
        Layer output;
        std::vector<Layer> hiddenLayers;
        Layer input;
        float learningRate;
};


#endif //NEURONAL_NETWORK_CPP_NETWORK_H