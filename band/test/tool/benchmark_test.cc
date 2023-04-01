#include "band/tool/benchmark.h"

#include <gtest/gtest.h>

namespace band {
namespace test {

TEST(BenchmarkTest, BenchmarkConfigLoadSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_TRUE(benchmark.Initialize(2, argv).ok());
}

TEST(BenchmarkTest, BenchmarkConfigLoadFail) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", ""};
  EXPECT_TRUE(!benchmark.Initialize(2, argv).ok());
  EXPECT_TRUE(!benchmark.Initialize(1, argv).ok());
}

TEST(BenchmarkTest, BenchmarkConfigRunSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_TRUE(benchmark.Initialize(2, argv).ok());
  EXPECT_TRUE(benchmark.Run().ok());
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