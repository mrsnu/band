#ifndef BAND_TEST_IMAGE_UTIL_H_
#define BAND_TEST_IMAGE_UTIL_H_

#include <memory>
#include <string>

namespace band {
class Buffer;
namespace test {

std::shared_ptr<Buffer> LoadImage(const std::string& filename);
std::tuple<unsigned char*, int, int> LoadRGBImageRaw(
    const std::string& filename);
void SaveImage(const std::string& filename, const Buffer& buffer);

}  // namespace test
}  // namespace band

#endif