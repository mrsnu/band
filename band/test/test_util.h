#ifndef BAND_TEST_TEST_UTIL_H_
#define BAND_TEST_TEST_UTIL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"

namespace Band {
namespace Test {

struct MockContextBase : public Context {
  MockContextBase() = default;

  MOCK_CONST_METHOD0(UpdateWorkerWaitingTime, void(void));
  MOCK_CONST_METHOD0(GetWorkerWaitingTime, const WorkerWaitingTime&(void));
  MOCK_CONST_METHOD0(GetIdleWorkers, std::set<WorkerId>(void));

  /* subgraph */
  MOCK_CONST_METHOD2(GetModelSubgraphKey, SubgraphKey(ModelId, WorkerId));
  MOCK_CONST_METHOD1(IsEnd, bool(const SubgraphKey&));
  MOCK_METHOD1(Invoke, BandStatus(const SubgraphKey&));

  /* model */
  MOCK_METHOD1(GetModelSpec, const ModelSpec*(ModelId));
  MOCK_CONST_METHOD1(GetModelConfigIdx, int(ModelId));
  MOCK_CONST_METHOD1(GetModelWorker, WorkerId(ModelId));

  /* scheduling */
  using WorkerWaiting = const std::map<WorkerId, int64_t>&;
  using ShortestLatencyWithUnitSubgraph =
      std::pair<std::vector<SubgraphKey>, int64_t>;
  using SubgraphWithShortestLatency =
      std::pair<std::vector<SubgraphKey>, int64_t>;
  MOCK_CONST_METHOD5(GetShortestLatency,
                     std::pair<SubgraphKey, int64_t>(int, std::set<int>,
                                                     int64_t, WorkerWaiting,
                                                     SubgraphKey));

  MOCK_CONST_METHOD3(GetShortestLatencyWithUnitSubgraph,
                     ShortestLatencyWithUnitSubgraph(int, int, WorkerWaiting));
  MOCK_CONST_METHOD2(GetSubgraphWithShortestLatency,
                     SubgraphWithShortestLatency(Job&, WorkerWaiting));
  MOCK_CONST_METHOD3(GetSubgraphIdxSatisfyingSLO,
                     SubgraphKey(Job&, WorkerWaiting,
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

  /* getters */
  ErrorReporter* GetErrorReporter() { return DefaultErrorReporter(); }
  MOCK_METHOD1(GetWorker, Worker*(WorkerId));

  /* tensor communication */
  MOCK_METHOD1(TryCopyInputTensors, BandStatus(const Job&));
  MOCK_METHOD1(TryCopyOutputTensors, BandStatus(const Job&));
};

}  // namespace Test
}  // namespace Band

#endif  // BAND_TEST_TEST_UTIL_H_