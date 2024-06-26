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

#ifndef BAND_TEST_TEST_UTIL_H_
#define BAND_TEST_TEST_UTIL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/engine_interface.h"

namespace band {
namespace test {

struct MockEngineBase : public IEngine {
  MockEngineBase() = default;

  MOCK_CONST_METHOD0(UpdateWorkersWaiting, void(void));
  MOCK_CONST_METHOD0(GetWorkerWaitingTime, WorkerWaitingTime(void));
  MOCK_CONST_METHOD0(GetIdleWorkers, std::set<WorkerId>(void));

  /* subgraph */
  MOCK_CONST_METHOD2(GetLargestSubgraphKey, SubgraphKey(ModelId, WorkerId));
  MOCK_CONST_METHOD1(IsBegin, bool(const SubgraphKey&));
  MOCK_CONST_METHOD1(IsEnd, bool(const SubgraphKey&));
  MOCK_CONST_METHOD1(HasSubgraph, bool(const SubgraphKey&));
  MOCK_CONST_METHOD1(ForEachSubgraph,
                     void(std::function<void(const SubgraphKey&)>));
  MOCK_METHOD1(Invoke, absl::Status(const SubgraphKey&));

  /* model */
  MOCK_CONST_METHOD1(GetModelSpec, const ModelSpec*(ModelId));
  MOCK_CONST_METHOD1(GetModelWorker, WorkerId(ModelId));

  /* scheduling */
  using WorkerWaiting = const std::map<WorkerId, int64_t>&;
  using ShortestLatencyWithUnitSubgraph =
      std::pair<std::vector<SubgraphKey>, int64_t>;
  using SubgraphWithShortestLatency =
      std::pair<std::vector<SubgraphKey>, int64_t>;
  MOCK_CONST_METHOD4(GetShortestLatency,
                     std::pair<SubgraphKey, int64_t>(ModelId, BitMask, int64_t,
                                                     WorkerWaiting));

  MOCK_CONST_METHOD3(GetShortestLatencyWithUnitSubgraph,
                     ShortestLatencyWithUnitSubgraph(ModelId, int,
                                                     WorkerWaiting));
  MOCK_CONST_METHOD2(GetSubgraphWithShortestLatency,
                     SubgraphWithShortestLatency(const Job&, WorkerWaiting));
  MOCK_CONST_METHOD3(GetSubgraphIdxSatisfyingSLO,
                     SubgraphKey(const Job&, WorkerWaiting,
                                 const std::set<WorkerId>&));

  /* profiler */
  MOCK_METHOD2(UpdateLatency, void(const SubgraphKey&, int64_t));
  MOCK_CONST_METHOD1(GetProfiled, int64_t(const SubgraphKey&));
  MOCK_CONST_METHOD1(GetExpected, int64_t(const SubgraphKey&));

  /* planner */
  MOCK_METHOD0(Trigger, void());
  MOCK_METHOD2(EnqueueRequest, JobId(Job, bool));
  MOCK_METHOD2(EnqueueBatch, std::vector<JobId>(std::vector<Job>, bool));
  MOCK_METHOD1(PrepareReenqueue, void(Job&));
  MOCK_METHOD1(EnqueueFinishedJob, void(Job&));
  MOCK_METHOD1(EnqueueToWorker, bool(const ScheduleAction&));
  MOCK_METHOD1(EnqueueToWorkerBatch, bool(const std::vector<ScheduleAction>&));

  /* getters */
  MOCK_METHOD1(GetWorker, Worker*(WorkerId));
  MOCK_CONST_METHOD1(GetWorker, const Worker*(WorkerId));
  MOCK_CONST_METHOD0(GetNumWorkers, size_t());

  /* tensor communication */
  MOCK_METHOD1(TryCopyInputTensors, absl::Status(const Job&));
  MOCK_METHOD1(TryCopyOutputTensors, absl::Status(const Job&));
};

}  // namespace test
}  // namespace band

#endif  // BAND_TEST_TEST_UTIL_H_