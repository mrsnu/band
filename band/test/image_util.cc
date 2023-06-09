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
  return Buffer::CreateFromRaw(
      data, width, height,
      num_channels == 1 ? BufferFormat::GrayScale : BufferFormat::RGB);
}

void SaveImage(const std::string& filename, const Buffer& buffer) {
  stbi_write_jpg(filename.c_str(), buffer.GetDimension()[0],
                 buffer.GetDimension()[1],
                 buffer.GetBufferFormat() == BufferFormat::GrayScale ? 1 : 3,
                 buffer[0].data, 100);
}

}  // namespace test
}  // namespace band