#include "model.hpp"

#include <fstream>

// TODO save image size in the model

void model::load(fs::path file_path, network& n)
{
    std::ifstream fin{ file_path, std::ifstream::in | std::ifstream::binary };
    fin.exceptions(std::ifstream::failbit);

    fin.read(reinterpret_cast<char*>(&n.learn_rate), sizeof(n.learn_rate));
    fin.read(reinterpret_cast<char*>(&n.input_neurons), sizeof(n.input_neurons));

    size_t hidden_neurons;
    size_t output_neurons;
    fin.read(reinterpret_cast<char*>(&hidden_neurons), sizeof(hidden_neurons));
    fin.read(reinterpret_cast<char*>(&output_neurons), sizeof(output_neurons));

    n.hidden.resize(hidden_neurons);
    n.output.resize(output_neurons);

    for (auto& h : n.hidden)
    {
        h.resize(n.input_neurons);
        fin.read(reinterpret_cast<char*>(h.data()), sizeof(double) * h.size());
    }

    for (auto& o : n.output)
    {
        o.resize(hidden_neurons);
        fin.read(reinterpret_cast<char*>(o.data()), sizeof(double) * o.size());
    }
}

void model::save(fs::path file_path, const network& n)
{
    std::ofstream fout{ file_path , std::ofstream::out | std::ofstream::binary };
    fout.exceptions(std::ofstream::failbit);

    fout.write(reinterpret_cast<const char*>(&n.learn_rate), sizeof(n.learn_rate));
    fout.write(reinterpret_cast<const char*>(&n.input_neurons), sizeof(n.input_neurons));

    size_t hidden_neurons = n.hidden.size();
    size_t output_neurons = n.output.size();
    fout.write(reinterpret_cast<const char*>(&hidden_neurons), sizeof(hidden_neurons));
    fout.write(reinterpret_cast<const char*>(&output_neurons), sizeof(output_neurons));

    for (auto& h : n.hidden)
    {
        for (auto w : h)
            fout.write(reinterpret_cast<const char*>(&w), sizeof(w));
    }

    for (auto& o : n.output)
    {
        for (auto w : o)
            fout.write(reinterpret_cast<const char*>(&w), sizeof(w));
    }
}
