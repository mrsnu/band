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

#include "band/logger.h"
#include "band/tool/benchmark.h"

using namespace band;

int main(int argc, const char** argv) {
  band::tool::Benchmark benchmark;
  if (benchmark.Initialize(argc, argv).ok()) {
    auto status = benchmark.Run();
    if (!status.ok()) {
      BAND_LOG(LogSeverity::kError, "Benchmark failed: %s", std::string(status.message()).c_str());
      return -1;
    }
  } else {
    BAND_LOG(LogSeverity::kError, "Benchmark failed to initialize");
    return -1;
  }
  return 0;
}
