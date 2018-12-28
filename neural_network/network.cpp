#include "network.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <string>

std::uniform_real_distribution<double> layer_dist(size_t input_size)
{
    double in = static_cast<double>(input_size);
    return std::uniform_real_distribution<double>{ -1.0 / std::sqrt(in), 1.0 / std::sqrt(in) };
}

network::network(size_t input_neurons, size_t hidden_neurons, size_t output_neurons, double learn_rate)
    : learn_rate{ learn_rate },
    input_neurons{ input_neurons }
{
    std::random_device rd;
    std::seed_seq seed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };

    std::mt19937_64 gen{ seed };

    hidden.resize(hidden_neurons);
    {
        auto dist = layer_dist(input_neurons);
        for (auto& h : hidden)
        {
            h.resize(input_neurons);
            std::generate(h.begin(), h.end(), [&] { return dist(gen); });
        }
    }

    output.resize(output_neurons);
    {
        auto dist = layer_dist(hidden_neurons);
        for (auto& o : output)
        {
            o.resize(hidden_neurons);
            std::generate(o.begin(), o.end(), [&] { return dist(gen); });
        }
    }
}

std::vector<double> network::predict(const std::vector<double>& input) const
{
    if (input.size() != input_neurons)
        throw invalid_input(input_neurons, input.size());

    // forward propagation algorithm

    std::vector<double> hidden_results;
    hidden_results.resize(hidden.size());

    // input to hidden
    std::transform(hidden.begin(), hidden.end(), hidden_results.begin(), [&](const auto& h)
                   {
                       return std::inner_product(h.begin(), h.end(), input.begin(), 0.0);
                   });
    // output from hidden
    std::transform(hidden_results.begin(), hidden_results.end(), hidden_results.begin(), activation);

    std::vector<double> out_result;
    out_result.resize(output.size());

    // input to outputs
    std::transform(output.begin(), output.end(), out_result.begin(), [&](const auto& o)
                   {
                       return std::inner_product(o.begin(), o.end(), hidden_results.begin(), 0.0);
                   });
    // final result
    std::transform(out_result.begin(), out_result.end(), out_result.begin(), activation);

    return out_result;
}

void network::train(const std::vector<double>& input, const std::vector<double>& target)
{
    if (input.size() != input_neurons)
        throw invalid_input(input_neurons, input.size());

    // not just calling predict(), as we need the intermediate results

    std::vector<double> hidden_out;
    hidden_out.resize(hidden.size());

    // input to hidden
    std::transform(hidden.begin(), hidden.end(), hidden_out.begin(), [&](const auto& h)
                   {
                       return std::inner_product(h.begin(), h.end(), input.begin(), 0.0);
                   });
     // output from hidden
    std::transform(hidden_out.begin(), hidden_out.end(), hidden_out.begin(), activation);

    std::vector<double> final_out;
    final_out.resize(output.size());

    // input to outputs
    std::transform(output.begin(), output.end(), final_out.begin(), [&](const auto& o)
                   {
                       return std::inner_product(o.begin(), o.end(), hidden_out.begin(), 0.0);
                   });
    // final result
    std::transform(final_out.begin(), final_out.end(), final_out.begin(), activation);

    // find errors
    std::vector<double> output_error;
    output_error.resize(output.size());

    std::vector<double> hidden_error;
    hidden_error.resize(hidden.size());

    std::transform(target.begin(), target.end(), final_out.begin(), output_error.begin(), std::minus<double>{});

    for (size_t i = 0; i < hidden_error.size(); i++)
    {
        hidden_error[i] = 0.0;
        for (size_t j = 0; j < output_error.size(); j++)
        {
            hidden_error[i] += output[j][i] * output_error[j];
        }
    }

    // adjust the weights accordingly (back propagation)
    back_propagate(output, hidden_out, final_out, output_error);
    back_propagate(hidden, input, hidden_out, hidden_error);
}

void network::back_propagate(std::vector<std::vector<double>>& layer, const std::vector<double>& layer_input, const std::vector<double>& layer_output, const std::vector<double>& layer_error)
{
    std::vector<double> mod_output;
    mod_output.resize(layer_output.size());
    std::transform(layer_output.begin(), layer_output.end(), layer_error.begin(), mod_output.begin(), [](double out, double error)
                   {
                       return error * activation_d(out);
                   });

    for (size_t i = 0; i < layer.size(); i++)
    {
        for (size_t j = 0; j < layer[i].size(); j++)
        {
            layer[i][j] += learn_rate * (mod_output[i] * layer_input[j]);
        }
    }
}

double network::activation(double x)
{
    return 1.0 / (1.0 + std::exp(-1.0 * x));
}

double network::activation_d(double x)
{
    return x * (1.0 - x);
}

invalid_input::invalid_input(size_t expected, size_t actual)
    : runtime_error("expected " + std::to_string(expected) + " inputs, actual: " + std::to_string(actual))
{}
