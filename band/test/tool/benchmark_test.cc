#include "band/tool/benchmark.h"

#include <gtest/gtest.h>

namespace Band {
namespace Test {

TEST(BenchmarkTest, BenchmarkConfigLoadSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), absl::OkStatus());
}

TEST(BenchmarkTest, BenchmarkConfigLoadFail) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", ""};
  EXPECT_EQ(!benchmark.Initialize(2, argv).ok());
  EXPECT_EQ(!benchmark.Initialize(1, argv).ok());
}

TEST(BenchmarkTest, BenchmarkConfigRunSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), absl::OkStatus());
  EXPECT_EQ(benchmark.Run(), absl::OkStatus());
}

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
#ifdef BAND_TFLITE
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif  // BAND_TFLITE
  return 0;
}