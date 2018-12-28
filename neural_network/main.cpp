#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <numeric>

#include "network.hpp"
#include "mnist.hpp"
#include "model.hpp"

namespace fs = std::filesystem;

enum class op
{
    train,
    predict,
};

size_t epochs = 5;
fs::path model_path;
fs::path label_path;
fs::path img_path;
op op;

struct fp_out {};
std::ostream& operator<<(std::ostream& os, fp_out);

void train();
void predict();
void parse_args(int argc, char** argv);

/*
usage:
    neural_network train [-e/--epochs #] [-m/--model file to output model to]
*/
int main(int argc, char** argv) try
{
    parse_args(argc, argv);

    switch (op)
    {
    case op::train:
        train();
        break;

    case op::predict:
        predict();
        break;
    }

    return 0;
}
catch (const std::ios_base::failure& e)
{
    std::cout << "io failure: " << e.code().value() << ' ' << e.code().message() << '\n';
    return 1;
}
catch (const std::exception& e)
{
    std::cout << e.what() << '\n';
    return 1;
}

void train()
{
    mnist::dataset data{ "mnist/train-labels-idx1-ubyte", "mnist/train-images-idx3-ubyte" };

    network net{ data.image_width() * data.image_height(), 200, 10, 0.1 };

    auto total_start = std::chrono::steady_clock::now();
    mnist::image input;
    for (size_t i = 0; i < 5; i++)
    {
        std::cout << "epoch #" << i << '\n';
        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t j = 0; j < data.size(); j++)
        {
            data.next(input);

            auto sample_start = std::chrono::steady_clock::now();
            net.train(input.pixels, input.label);
            auto sample_end = std::chrono::steady_clock::now();
        }

        auto epoch_end = std::chrono::steady_clock::now();
        std::cout << "length " << std::chrono::duration_cast<std::chrono::duration<double>>(epoch_end - epoch_start).count() << "sec\n";

        data.reset();
    }
    auto total_end = std::chrono::steady_clock::now();

    std::cout << "total time: " << std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count() << "sec\n";

    if (model_path.empty())
        model_path = std::to_string(total_end.time_since_epoch().count()) + ".model";
    model::save(model_path, net);
    std::cout << "wrote " << model_path.string() << '\n';
}

void predict()
{
    if (model_path.empty())
        throw std::runtime_error{ "missing model path argument (required for predicting)" };

    mnist::dataset data{ "mnist/t10k-labels-idx1-ubyte", "mnist/t10k-images-idx3-ubyte" };

    network net;
    model::load(model_path, net);

    std::array<size_t, 10> correct_guesses;
    correct_guesses.fill(0);
    std::array<size_t, 10> actual_amounts;
    actual_amounts.fill(0);

    mnist::image input;
    for (size_t i = 0; i < data.size(); i++)
    {
        data.next(input);
        std::vector<double> out = net.predict(input.pixels);

        auto actual = std::distance(input.label.begin(), std::max_element(input.label.begin(), input.label.end()));
        auto guessed = std::distance(out.begin(), std::max_element(out.begin(), out.end()));

        if (actual == guessed)
            correct_guesses[actual]++;
        actual_amounts[actual]++;
    }

    size_t total_correct = std::accumulate(correct_guesses.begin(), correct_guesses.end(), 0ull);
    double total_percentage = (static_cast<double>(total_correct) / static_cast<double>(data.size())) * 100.0;
    std::cout << total_correct << " / " << data.size();
    std::cout << " - " << fp_out{} << total_percentage << "% correct\n\n";

    std::cout << "per label:\n";
    for (size_t i = 0; i < 10; i++)
    {
        double percentage = (static_cast<double>(correct_guesses[i]) / static_cast<double>(actual_amounts[i])) * 100.0;
        std::cout << "'" << i << "' - " << std::right << std::setw(4) << correct_guesses[i] << " / " << std::right << std::setw(4) << actual_amounts[i];
        std::cout << " - " << fp_out{} << percentage << "%\n";
    }
}

std::ostream& operator<<(std::ostream& os, fp_out)
{
    os << std::setw(6) << std::right << std::fixed << std::setprecision(2);
    return os;
}

void parse_args(int argc, char** argv)
{
    std::vector<std::string> args;
    args.assign(argv + 1, argv + argc);

    if (args.empty())
        throw std::runtime_error{ "no operation specified. must be 'train' or 'predict'" };

    if (args[0] == "train")
        op = op::train;
    else if (args[0] == "predict")
        op = op::predict;

    for (int i = 1; i < args.size(); i++)
    {
        if (args[i] == "-e" || args[i] == "--epochs")
        {
            i++;
            if (i >= args.size())
                throw std::runtime_error{ "missing value for epochs argument" };

            epochs = std::stoul(args[i]);
        }
        else if (args[i] == "-m" || args[i] == "--model")
        {
            i++;
            if (i >= args.size())
                throw std::runtime_error{ "missing value for model argument" };

            model_path = args[i];
        }
        else if (args[i] == "-l" || args[i] == "--labels")
        {
            i++;
            if (i >= args.size())
                throw std::runtime_error{ "missing value for labels argument" };

            label_path = args[i];
        }
        else if (args[i] == "-i" || args[i] == "--images")
        {
            i++;
            if (i >= args.size())
                throw std::runtime_error{ "missing value for images argument" };

            img_path = args[i];
        }
    }
}
