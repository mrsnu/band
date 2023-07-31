#include "band/tool/benchmark.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(BenchmarkTest, BenchmarkConfigLoadSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), absl::OkStatus());
}

TEST(BenchmarkTest, BenchmarkConfigLoadFail) {
  // intentionally silent the log, to avoid the test failure due to the error
  // log
  Logger::SetVerbosity(BAND_LOG_NUM_SEVERITIES);
  tool::Benchmark benchmark;
  const char* argv[] = {"", ""};
  EXPECT_FALSE(benchmark.Initialize(2, argv).ok());
  EXPECT_FALSE(benchmark.Initialize(1, argv).ok());
}

TEST(BenchmarkTest, BenchmarkConfigRunSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), absl::OkStatus());
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