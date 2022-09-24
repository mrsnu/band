#ifndef BAND_TIME_H_
#define BAND_TIME_H_

#include <cstdint>

namespace Band {
namespace Time {
uint64_t NowMicros();
uint64_t NowNanos();
void SleepForMicros(uint64_t micros);
} // namespace Time
} // namespace Band
#endif // BAND_TIME_H_
