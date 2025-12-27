#include <iostream>

#include "network/Layer.h"

int main() {
    Layer layer(LeakyReLU,3,2);
    Vector input = Vector::ones(2);
    layer.activate(input);
    return 0;
}
