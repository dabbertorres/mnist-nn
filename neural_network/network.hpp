#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

struct network
{
    network() = default;
    network(size_t input_neurons, size_t hidden_neurons, size_t output_neurons, double learn_rate);

    double learn_rate;
    size_t input_neurons;
    std::vector<std::vector<double>> hidden; // single hidden layer
    std::vector<std::vector<double>> output;

    std::vector<double> predict(const std::vector<double>& input) const;
    void train(const std::vector<double>& input, const std::vector<double>& target);

    // just a simple sigmoid function
    static double activation(double x);
    static double activation_d(double x);

protected:
    void back_propagate(std::vector<std::vector<double>>& layer, const std::vector<double>& layer_input, const std::vector<double>& layer_output, const std::vector<double>& layer_error);
};

struct invalid_input : public std::runtime_error
{
    invalid_input(size_t expected, size_t actual);
};
