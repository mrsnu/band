#ifndef BAND_TIME_H_
#define BAND_TIME_H_

#include <cstdint>

namespace band {
namespace time {
uint64_t NowMicros();
uint64_t NowNanos();
void SleepForMicros(uint64_t micros);
}  // namespace time
}  // namespace band
#endif  // BAND_TIME_H_
