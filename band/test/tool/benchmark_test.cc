#include "band/tool/benchmark.h"

#include <gtest/gtest.h>

namespace Band {
namespace Test {

TEST(BenchmarkTest, BenchmarkConfigLoadSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), kBandOk);
}

TEST(BenchmarkTest, BenchmarkConfigLoadFail) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", ""};
  EXPECT_EQ(benchmark.Initialize(2, argv), kBandError);
  EXPECT_EQ(benchmark.Initialize(1, argv), kBandError);
}

TEST(BenchmarkTest, BenchmarkConfigRunSuccess) {
  tool::Benchmark benchmark;
  const char* argv[] = {"", "band/test/data/benchmark_config.json"};
  EXPECT_EQ(benchmark.Initialize(2, argv), kBandOk);
  EXPECT_EQ(benchmark.Run(), kBandOk);
}

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}