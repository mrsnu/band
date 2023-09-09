// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  // intentionally silent the log, to avoid the test failure due to the error log
  Logger::Get().SetVerbosity(LogSeverity::kError);
  tool::Benchmark benchmark;
  const char* argv[] = {"", ""};
  EXPECT_TRUE(!benchmark.Initialize(2, argv).ok());
  EXPECT_TRUE(!benchmark.Initialize(1, argv).ok());
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