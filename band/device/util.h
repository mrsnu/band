#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <string>
#include <vector>

#ifdef __ANDROID__
#define BAND_SUPPORT_DEVICE 1
#else
#define BAND_SUPPORT_DEVICE 0
#endif

namespace band {

int TryReadInt(std::vector<std::string> paths);
std::vector<int> TryReadInts(std::vector<std::string> paths);
std::string TryReadString(std::vector<std::string> paths);
bool IsRooted();

}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
