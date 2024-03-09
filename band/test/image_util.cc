#include "band/test/image_util.h"

#include "band/buffer/buffer.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace band {
namespace test {

std::shared_ptr<Buffer> LoadImage(const std::string& filename) {
  int width, height, num_channels;
  unsigned char* data =
      stbi_load(filename.c_str(), &width, &height, &num_channels, 0);

  if (data == nullptr) {
    return nullptr;
  }
  return std::shared_ptr<Buffer>(Buffer::CreateFromRaw(
      data, width, height,
      num_channels == 1 ? BufferFormat::kGrayScale : BufferFormat::kRGB));
}

std::tuple<unsigned char*, int, int> LoadRGBImageRaw(
    const std::string& filename) {
  int width, height, num_channels;
  unsigned char* data =
      stbi_load(filename.c_str(), &width, &height, &num_channels, 0);

  if (data == nullptr) {
    return std::make_tuple(nullptr, 0, 0);
  }
  return std::make_tuple(data, width, height);
}

void SaveImage(const std::string& filename, const Buffer& buffer) {
  stbi_write_jpg(filename.c_str(), buffer.GetDimension()[0],
                 buffer.GetDimension()[1],
                 buffer.GetBufferFormat() == BufferFormat::kGrayScale ? 1 : 3,
                 buffer[0].data, 100);
}

}  // namespace test
}  // namespace band