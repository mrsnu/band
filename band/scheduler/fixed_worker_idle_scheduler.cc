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

#include "band/scheduler/fixed_worker_idle_scheduler.h"

#include "band/logger.h"

namespace band {
bool FixedWorkerIdleScheduler::Schedule(JobQueue& requests) {
  bool success = true;
  while (!requests.empty()) {
    Job to_execute = requests.front();
    requests.pop_front();

    if (to_execute.target_worker_id == -1) {
      BAND_LOG_PROD(BAND_LOG_ERROR,
                    "Job target_worker_id is not set. Fallback to 0.");
      to_execute.target_worker_id = 0;
    }

    int model_id = to_execute.model_id;
    SubgraphKey key =
        engine_.GetLargestSubgraphKey(model_id, to_execute.target_worker_id);
    int fixed_idle_us = engine_.GetExperimentConfig().fixed_idle_us;
    success &= engine_.EnqueueToWorker({to_execute, key}, fixed_idle_us);
  }
  return success;
}

}  // namespace band
