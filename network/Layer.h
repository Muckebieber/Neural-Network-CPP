//
// Created by martin on 26.12.25.
//

#ifndef NEURONAL_NETWORK_CPP_LAYER_H
#define NEURONAL_NETWORK_CPP_LAYER_H
#include "../math/Vector.h"
#include "../math/Matrix.h"
#include  "../enums/ErrorFunction.h"
#include  "../enums/ActivationFunction.h"

class Layer {
    public:
        Layer(ActivationFunction activationFunction, int nodes , int nodesPrev) : activationFunction(activationFunction), A(nodes), Z(nodes), W(nodes,nodesPrev), X(nodesPrev), B(nodes), D(nodes), G(nodes,nodesPrev) {};
        Vector activate(Vector &X);
        Vector calcDelta(Vector &classes, ErrorFunction errorFunction);
        Vector calcDelta(Layer& layer);
        Matrix calcGradients();
        void adaptWeights(const Matrix& gradients, float learningRate);
        const ActivationFunction getActivationFunction() const {return this->activationFunction;}
        const Vector& getA() const {return this->A;}
        const Vector& getZ() const {return this->Z;}
        const Matrix& getW() const {return this->W;}
        const Vector& getX() const {return this->X;}
        const Vector& getB() const {return this->B;}
        const Vector& getD() const {return this->D;}
        const Matrix& getG() const {return this->G;}
    private:
        ActivationFunction activationFunction;
        Vector A;
        Vector Z;
        Matrix W;
        Vector X;
        Vector B;
        Vector D;
        Matrix G;
};


#endif //NEURONAL_NETWORK_CPP_LAYER_H