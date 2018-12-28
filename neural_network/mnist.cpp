#include "mnist.hpp"

// the file formats use big-endian ordering
#if defined(__GNUC__) || defined(__clang__)
#define BYTESWAP(x) __builtin_bswap32(static_cast<int32_t>(x))
#elif defined(_MSC_VER)
#define BYTESWAP(x) _byteswap_ulong(static_cast<unsigned long>(x))
#endif

namespace mnist
{

struct label_file_header
{
    static constexpr uint32_t expected_magic = 0x0000'0801;

    uint32_t magic;
    uint32_t num_items;

    void byteswap() noexcept
    {
        magic = BYTESWAP(magic);
        num_items = BYTESWAP(num_items);
    }

    bool valid() const noexcept
    {
        return magic == expected_magic;
    }
};

struct image_file_header
{
    static constexpr uint32_t expected_magic = 0x0000'0803;

    uint32_t magic;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;

    void byteswap() noexcept
    {
        magic = BYTESWAP(magic);
        num_items = BYTESWAP(num_items);
        num_rows = BYTESWAP(num_rows);
        num_cols = BYTESWAP(num_cols);
    }

    bool valid() const noexcept
    {
        return magic == expected_magic;
    }
};

dataset::dataset(fs::path label_path, fs::path image_path)
    : _label_file{ label_path, std::ifstream::in | std::ifstream::binary },
    _image_file{ image_path, std::ifstream::in | std::ifstream::binary }
{
    _label_file.exceptions(std::ifstream::failbit);
    _image_file.exceptions(std::ifstream::failbit);

    label_file_header label_header;
    image_file_header image_header;

    _label_file.read(reinterpret_cast<char*>(&label_header), sizeof(label_header));
    _image_file.read(reinterpret_cast<char*>(&image_header), sizeof(image_header));

    label_header.byteswap();
    image_header.byteswap();

    if (!label_header.valid())
        throw file_header_exception{ label_path, "magic value", "expected " + std::to_string(label_file_header::expected_magic) + ", but got " + std::to_string(label_header.magic) };

    if (!image_header.valid())
        throw file_header_exception{ image_path, "magic value", "expected " + std::to_string(image_file_header::expected_magic) + ", but got " + std::to_string(image_header.magic) };

    if (label_header.num_items != image_header.num_items)
        throw file_header_exception{ label_path, "different number of items", std::to_string(label_header.num_items) + " != " + std::to_string(image_header.num_items) };

    _size = label_header.num_items;
    _img_width = image_header.num_cols;
    _img_height = image_header.num_rows;
}

void dataset::next(image& img)
{
    uint8_t label;
    _label_file.read(reinterpret_cast<char*>(&label), 1);

    // try to reduce needed allocations.
    static std::vector<uint8_t> pixels;
    pixels.resize(_img_width * _img_height);
    _image_file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

    img.label.clear();
    img.label.resize(10);

    img.pixels.clear();
    img.pixels.resize(_img_width * _img_height);

    img.label[label] = 1.0;
    std::transform(pixels.begin(), pixels.end(), img.pixels.begin(), [](uint8_t v) { return static_cast<double>(v) / 255.0; });
}

void dataset::reset()
{
    _label_file.seekg(sizeof(label_file_header), std::ifstream::beg);
    _image_file.seekg(sizeof(image_file_header), std::ifstream::beg);
}

size_t dataset::size() const noexcept
{
    return _size;
}

size_t dataset::image_width() const noexcept
{
    return _img_width;
}

size_t dataset::image_height() const noexcept
{
    return _img_height;
}

file_header_exception::file_header_exception(fs::path file, std::string what, std::string why)
    : std::runtime_error("file header ('" + file.string() + "'): " + what + " - " + why)
{}

}
