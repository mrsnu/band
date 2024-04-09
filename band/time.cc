#include "band/time.h"

#if defined(_MSC_VER)
#include <chrono>  // NOLINT(build/c++11)
#include <thread>  // NOLINT(build/c++11)
#else
#include <sys/time.h>
#include <time.h>
#endif

namespace band {
/**
 * @namespace time
 * @brief Contains functions for time-related operations.
 */
namespace time {

#if defined(_MSC_VER)

/**
 * @brief Returns the current time in microseconds since the epoch.
 * @return The current time in microseconds.
 */
uint64_t NowMicros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/**
 * @brief Returns the current time in nanoseconds since the epoch.
 * @return The current time in nanoseconds.
 */
uint64_t NowNanos() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/**
 * @brief Sleeps for the specified number of microseconds.
 * @param micros The number of microseconds to sleep for.
 */
void SleepForMicros(uint64_t micros) {
  std::this_thread::sleep_for(std::chrono::microseconds(micros));
}

#else

/**
 * @brief Returns the current time in microseconds since the epoch.
 * @return The current time in microseconds.
 */
uint64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64_t>(tv.tv_sec) * 1e6 + tv.tv_usec;
}

/**
 * @brief Returns the current time in nanoseconds since the epoch.
 * @return The current time in nanoseconds.
 */
uint64_t NowNanos() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1e9 + ts.tv_nsec;
}

/**
 * @brief Sleeps for the specified number of microseconds.
 * @param micros The number of microseconds to sleep for.
 */
void SleepForMicros(uint64_t micros) {
  timespec sleep_time;
  sleep_time.tv_sec = micros / 1e6;
  micros -= sleep_time.tv_sec * 1e6;
  sleep_time.tv_nsec = micros * 1e3;
  nanosleep(&sleep_time, nullptr);
}

#endif  // defined(_MSC_VER)

}  // namespace time
}  // namespace band
