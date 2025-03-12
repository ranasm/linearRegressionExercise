#include "utils.h"
#include <cmath>
#include <numeric>

const int NUM_FEATURES = 2;
const int DEFAULT_FEATURE_VALUE = 1;
const int DEFAULT_WEIGHT_VALUE = 0;

std::vector<float> convertToFeatureVector(int x) {
    // [1, x]
    std::vector<float> featureVector(NUM_FEATURES,DEFAULT_FEATURE_VALUE);
    featureVector.at(NUM_FEATURES-1) = x;
    return featureVector;
}

std::vector<float> initializeWeights() {
    // Init weights to [0, 0]
    std::vector<float> weights(NUM_FEATURES,DEFAULT_WEIGHT_VALUE);
    return weights;
}

float calculateTrainingLoss(std::vector<float> weights, std::vector<std::vector<float>> trainingData) {
    float totalLoss = 0.0;
    float scaleFactor = float(1) / float((trainingData.size()));
    for (int i = 0; i < trainingData.size(); i++) {
        std::vector<float> xyDataPair = trainingData.at(i);
        float x = xyDataPair.at(0);
        float y = xyDataPair.at(1);

        std::vector<float> featureVector = convertToFeatureVector(x);

        // Calculate score (in otherwords, calculate prediction (ie h(x) = w1*x1 + w2*x2 + ... + wn*xn)
        double score = inner_product(weights.begin(), weights.end(), featureVector.begin(), 0.0);

        double residual = y - score;
        double loss = pow(residual, 2);
        totalLoss += loss;
    }
    totalLoss *= scaleFactor;
    return totalLoss;
}

// Function to add two vectors and store the result in the first vector
void addVectors(std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        vec1[i] += vec2[i];
    }
}

std::vector<float> calculateGradient(std::vector<float> weights, std::vector<std::vector<float>> trainingData) {
    std::vector<float> gradient(NUM_FEATURES,DEFAULT_WEIGHT_VALUE);
    float scaleFactor = float(1) / float((trainingData.size()));
    for (int i = 0; i < trainingData.size(); i++) {
        std::vector<float> xyDataPair = trainingData.at(i);
        float x = xyDataPair.at(0);
        float y = xyDataPair.at(1);

        std::vector<float> featureVector = convertToFeatureVector(x);

        // Calculate score (in otherwords, calculate prediction (ie h(x) = w1*x1 + w2*x2 + ... + wn*xn)
        double score = inner_product(weights.begin(), weights.end(), featureVector.begin(), 0.0);

        double residual = (score - y)*2;

        for (auto& element : featureVector) {
            element *= residual;
        }
        addVectors(gradient, featureVector);
    }

    // Take total gradient and scale it by 1/N
    for (auto& element : gradient) {
        element *= scaleFactor;
    }
    return gradient;    
}

void runGradientDescent(std::vector<std::vector<float>> trainingData, float learningRate, int numIterations) {
    std::vector<float> weights = initializeWeights();
    float alpha = learningRate;
    for (int i = 0; i < numIterations; i++) {
        float loss = calculateTrainingLoss(weights, trainingData);
        std::vector<float> gradient = calculateGradient(weights, trainingData);
        for (int j = 0; j < weights.size(); j++) {
            weights.at(j) -= (alpha * gradient.at(j));
        }

        // Print metrics
        std::cout << "Iteration: " << i << " Loss: " << loss << " Weights: ";
        for (auto& weight : weights) {
            std::cout << weight << " ";
        }
        std::cout << "Gradient: ";
        for (auto& grad : gradient) {
            std::cout << grad << " ";
        }
        std::cout << std::endl;
    }
}

