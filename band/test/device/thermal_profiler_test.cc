#include "band/profiler/thermal_profiler.h"

#include "band/test/device/device_util.h"

#include <thread>

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(BandTestThermalProfilerTest, ThermalProfilerTest) {
  ThermalProfiler profiler(GetPixel4DeviceConfig());

  auto start_time = std::chrono::system_clock::now();
  while (true) {
    if (std::chrono::system_clock::now() - start_time >
        std::chrono::seconds(10 * 60)) {
      break;
    }
    auto handle = profiler.BeginEvent();
    profiler.EndEvent(handle);

    auto interval = profiler.GetInterval(handle);
    auto interval_time = interval.second.first - interval.first.first;
    auto interval_thermal = interval.second.second;
    for (auto pair : interval_thermal) {
      auto sensor = pair.first;
      auto temp = pair.second;
      std::cout << ToString(sensor) << ": " << temp << std::endl;
    }
    std::cout << "Interval time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     interval_time)
                     .count()
              << " ms" << std::endl;
    std::cout << "Size: " << profiler.GetNumEvents() << std::endl;
  }
}

}  // namespace test

}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
