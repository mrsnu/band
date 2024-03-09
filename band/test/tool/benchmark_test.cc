#include "band/tool/benchmark.h"

#include <gtest/gtest.h>

namespace band {
namespace test {
std::string severity = std::to_string(BAND_LOG_NUM_SEVERITIES);

TEST(BenchmarkTest, BenchmarkConfigLoadSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json",
                        severity.c_str()};
  EXPECT_EQ(benchmark.Initialize(3, argv), absl::OkStatus());
}

TEST(BenchmarkTest, BenchmarkConfigLoadFail) {
  // intentionally silent the log, to avoid the test failure due to the error
  // log
  tool::Benchmark benchmark;
  const char* argv[] = {"", "", severity.c_str()};
  EXPECT_TRUE(!benchmark.Initialize(3, argv).ok());
  EXPECT_TRUE(!benchmark.Initialize(2, argv).ok());
}

TEST(BenchmarkTest, BenchmarkConfigRunSuccess) {
  tool::Benchmark benchmark;
  std::string severity = std::to_string(BAND_LOG_NUM_SEVERITIES);
  const char* argv[] = {"", "band/test/data/benchmark_config.json",
                        severity.c_str()};
  EXPECT_EQ(benchmark.Initialize(3, argv), absl::OkStatus());
  EXPECT_EQ(benchmark.Run(), absl::OkStatus());
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}