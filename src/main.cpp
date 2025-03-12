#include <iostream>
#include "utils.h"

int main() {
    std::cout << "Implement linear regression using Gradient Descent!" << std::endl;

    std::vector<std::vector<float>> trainingData = {
        {1, 2},
        {2, 3},
        {4, 3},
    };

    float learningRate = 0.1;
    int numIterations = 500;

    runGradientDescent(trainingData, learningRate, numIterations);

    return 0;
}