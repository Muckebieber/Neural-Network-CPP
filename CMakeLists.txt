cmake_minimum_required(VERSION 4.2)
project(Neuronal_Network_CPP CUDA)

set(CMAKE_CUDA_STANDARD 20)
include_directories(${PROJECT_SOURCE_DIR}/math)
include_directories(${PROJECT_SOURCE_DIR}/network)
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/enums)
add_executable(Neuronal_Network_CPP main.cu
        network/Network.cu
        network/Network.h
        network/Layer.cu
        network/Layer.h
        enums/ErrorFunction.h
        enums/ActivationFunction.h
        math/Vector.cu
        math/Vector.h
        math/Matrix.cu
        math/Matrix.h
        utils/NeuralNetworkUtils.cu
        utils/NeuralNetworkUtils.h)

set_target_properties(Neuronal_Network_CPP PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)