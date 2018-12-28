#pragma once

#include <filesystem>
#include <vector>

#include "network.hpp"

namespace model
{
    namespace fs = std::filesystem;

    void load(fs::path file_path, network& n);
    void save(fs::path file_path, const network& n);
}
