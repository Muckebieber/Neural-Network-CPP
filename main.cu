#include <iostream>

#include "network/Layer.h"

int main() {
    Layer layer(LeakyReLU,3,2);
    Vector input = Vector::ones(2);
    std::cout << layer.activate(input).toString() << std::endl;
    return 0;
}
