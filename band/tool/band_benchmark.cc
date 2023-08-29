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
  auto status = benchmark.Initialize(argc, argv);
  if (status.ok()) {
    auto status = benchmark.Run();
    if (!status.ok()) {
      BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed: %s", status.message());
      return -1;
    }
  } else {
    BAND_LOG_PROD(BAND_LOG_ERROR, "Benchmark failed to initialize: %s", status.message());
    return -1;
  }
  return 0;
}
