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

#include "band/scheduler/round_robin_scheduler.h"

#include <algorithm>

namespace band {

bool RoundRobinScheduler::Schedule(JobQueue& requests) {
  std::set<WorkerId> idle_workers = engine_.GetIdleWorkers();
  bool success = true;

  for (auto worker_id : idle_workers) {
    if (!requests.empty()) {
      auto available_job = std::find_if(
          requests.begin(), requests.end(), [this, worker_id](const Job& job) {
            return engine_.GetLargestSubgraphKey(job.model_id, worker_id)
                .IsValid();
          });
      if (available_job != requests.end()) {
        Job to_execute = *available_job;
        SubgraphKey key =
            engine_.GetLargestSubgraphKey(to_execute.model_id, worker_id);
        success &= engine_.EnqueueToWorker({to_execute, key});
        requests.erase(available_job);
      }
    }
  }

  return success;
}

}  // namespace band
