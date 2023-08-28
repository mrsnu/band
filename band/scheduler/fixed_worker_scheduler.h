/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_
#define BAND_SCHEDULER_FIXED_WORKER_SCHEDULER_H_

#include "band/scheduler/scheduler.h"

namespace band {

// Assigns requested model to devices according to a direct request from engine
// or model_id.
class FixedWorkerScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  bool NeedProfile() override { return false; }
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kDeviceQueue; }
};

class FixedWorkerGlobalQueueScheduler : public IScheduler {
 public:
  using IScheduler::IScheduler;
  bool Schedule(JobQueue& requests) override;
  // Required for checking SLO violation.
  // We could add an option to this planner for skipping the SLO check,
  // in which case this function can return false.
  bool NeedProfile() override { return true; }
  bool NeedFallbackSubgraphs() override { return false; }
  WorkerType GetWorkerType() override { return WorkerType::kGlobalQueue; }
};

}  // namespace band

#endif  // BAND_SCHEDULER_fixed_worker_scheduler_H_
