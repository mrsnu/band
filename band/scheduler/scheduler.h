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

#ifndef BAND_SCHEDULER_SCHEDULER_H_
#define BAND_SCHEDULER_SCHEDULER_H_

#include <map>

#include "band/engine_interface.h"

namespace band {
class Planner;

class IScheduler {
 public:
  explicit IScheduler(IEngine& engine) : engine_(engine) {}
  virtual ~IScheduler() = default;
  // A Schedule() function is expected to do the followings:
  // For the given requests, selected requests to schedule and
  // find the appropriate devices. The selected requests should be
  // enqueued to the worker and removed from original queue.
  // Returns false if the scheduler wants to be called again.
  virtual bool Schedule(JobQueue& requests) = 0;
  virtual bool NeedFallbackSubgraphs() = 0;
  virtual WorkerType GetWorkerType() = 0;

 protected:
  IEngine& engine_;
};
}  // namespace band

#endif