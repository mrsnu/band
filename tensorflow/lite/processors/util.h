#ifndef TENSORFLOW_LITE_PROCESSORS_UTIL_H_
#define TENSORFLOW_LITE_PROCESSORS_UTIL_H_

#include <string>
#include <vector>

namespace tflite {
namespace impl {

int TryReadInt(std::vector<std::string> paths);
std::vector<int> TryReadInts(std::vector<std::string> paths);
std::string TryReadString(std::vector<std::string> paths);
bool IsRooted();

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROCESSORS_UTIL_H_
