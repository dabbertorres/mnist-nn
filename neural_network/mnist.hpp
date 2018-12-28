#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mnist
{

namespace fs = std::filesystem;

struct image
{
    // label contains 10 values, 1 of which will be 1.0, with rest as 0.0.
    std::vector<double> label;

    // pixels contains pixel values scaled to 0..1 from 0..255
    std::vector<double> pixels;
};

class dataset
{
public:
    dataset() = default;
    dataset(fs::path label_path, fs::path image_path);
    ~dataset() = default;

    // assigns img to the next entry in the data set.
    void next(image& img);

    // starts from the beginning data set
    void reset();

    size_t size() const noexcept;
    size_t image_width() const noexcept;
    size_t image_height() const noexcept;

private:
    std::ifstream _label_file;
    std::ifstream _image_file;

    size_t _size;
    size_t _img_width;
    size_t _img_height;
};

struct file_header_exception : public std::runtime_error
{
    file_header_exception(fs::path file, std::string what, std::string why);
};

}
