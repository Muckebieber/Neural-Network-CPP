#include <iostream>

#include "Network.h"
#include "network/Layer.h"

int main() {
    Network net(2,1,5,2,LeakyReLU,Sigmoid,MSE,0.01);
    // AND-Gatter Inputs
    std::vector<Vector> features;
    features.resize(4, Vector(2)); // 4 Inputs, je 2 Elemente
    features[0][0] = 0.0; features[0][1] = 0.0;
    features[1][0] = 0.0; features[1][1] = 1.0;
    features[2][0] = 1.0; features[2][1] = 0.0;
    features[3][0] = 1.0; features[3][1] = 1.0;

    // AND-Gatter Outputs / Klassen
    std::vector<Vector> classes;
    classes.resize(4, Vector(1)); // 4 Outputs, je 1 Element
    classes[0][0] = 0.0;
    classes[1][0] = 0.0;
    classes[2][0] = 0.0;
    classes[3][0] = 1.0;

    net.train(features,classes,10,1);
    return 0;
}
