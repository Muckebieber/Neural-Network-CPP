//
// Created by martin on 26.12.25.
//

#include "Network.h"

Network::Network(size_t inputs, size_t outputs, size_t hiddenNeurons, size_t hiddenLayers, ActivationFunction hiddenActivation, ActivationFunction outputActivaton, ErrorFunction eFunc, float learningRate) :input(hiddenActivation, hiddenNeurons, inputs), output(outputActivaton, outputs, hiddenNeurons), errorFunction(eFunc), gradients(hiddenLayers+2), deltas(hiddenLayers+2), learningRate(learningRate){
    for (int i = 0; i < hiddenLayers - 1; ++i) {
        this->hiddenLayers.emplace_back(hiddenActivation, hiddenNeurons, hiddenNeurons);
    }
}

Vector Network::predict(const Vector &X) {
    Vector X_PREV = input.activate(X);
    for (Layer &layer : this->hiddenLayers) {
        X_PREV = layer.activate(X_PREV);
    }
    return output.activate(X_PREV);
}

void Network::backpropagate(const Vector &X, const Vector &O_True) {

    this->predict(X);

    this->deltas[deltas.size()-1] += this->output.calcDelta(O_True, this->errorFunction);

    Layer& LAYER_PREV = this->output;
    for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
        this->deltas[i+1] += hiddenLayers[i].calcDelta(LAYER_PREV);
        LAYER_PREV = hiddenLayers[i];
    }
    this->deltas[0] += this->input.calcDelta(LAYER_PREV);

    this->gradients[0] += this->input.calcGradients();
    for (int i = 0; i < hiddenLayers.size(); ++i) {
        this->gradients[i+1] += this->hiddenLayers[i].calcGradients();
    }
    this->gradients[hiddenLayers.size()+1] += this->output.calcGradients();
}

void Network::train(const std::vector<Vector> &features, const std::vector<Vector> &classes, int iterations, int batchSize) {

    for (int i = 0; i < iterations; ++i) {

        for (int j = 0; j < features.size()/batchSize; ++j) {

            std::vector<Vector> miniBatchFeatures(batchSize);
            std::vector<Vector> miniBatchClasses(batchSize);

            for (int k = 0; k < batchSize; ++k) {
                miniBatchFeatures[k] = features[j * batchSize + k];
                miniBatchClasses[k] = classes[j * batchSize + k];
            }
            size_t x = 0;
            for (Vector &miniFeature : miniBatchFeatures) {
                this->backpropagate(miniFeature,classes[x]);
                x++;
            }

            for (Matrix &gradient : gradients) {
                gradient /= miniBatchFeatures.size();
            }

            for (Vector &delta : deltas) {
                delta /= miniBatchClasses.size();
            }

            adaptParameters(this->gradients, this->deltas);
        }

    }

}

void Network::adaptParameters(const std::vector<Matrix> &gradients, const std::vector<Vector> &deltas) {
    this->input.adaptWeights(gradients[0],this->learningRate);
    this->input.adaptBiases(deltas[0], this->learningRate);
    for (int i = 0; i < hiddenLayers.size(); ++i) {
        hiddenLayers[i].adaptWeights(gradients[i+1],this->learningRate);
        hiddenLayers[i].adaptBiases(deltas[i+1], this->learningRate);
    }
    this->output.adaptWeights(gradients[hiddenLayers.size()+1],this->learningRate);
    this->output.adaptBiases(deltas[hiddenLayers.size()+1], this->learningRate);
}
