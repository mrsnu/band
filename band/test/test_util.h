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
  using WorkerWaiting = const std::map<WorkerId, double>&;
  using MinCostWithUnitSubgraph = std::pair<std::vector<SubgraphKey>, double>;
  using SubgraphWithMinCost = std::pair<std::vector<SubgraphKey>, double>;

  MOCK_CONST_METHOD4(GetMinCostWithUnitSubgraph,
                     MinCostWithUnitSubgraph(ModelId, int, WorkerWaiting,
                                             const CostFunc));

  MOCK_CONST_METHOD3(GetSubgraphWithMinCost,
                     SubgraphWithMinCost(const Job&, WorkerWaiting,
                                         const CostFunc));
  MOCK_CONST_METHOD3(GetSubgraphIdxSatisfyingSLO,
                     SubgraphKey(const Job&, const WorkerWaitingTime&,
                                 const std::set<WorkerId>&));

  /* estimators */
  MOCK_METHOD2(Update, void(const SubgraphKey&, int64_t));
  MOCK_METHOD2(UpdateWithJob, void(const SubgraphKey&, Job&));
  MOCK_CONST_METHOD1(GetProfiled, double(const SubgraphKey&));
  MOCK_CONST_METHOD1(GetExpected, double(const SubgraphKey&));

  /* profiler */
  MOCK_METHOD0(BeginEvent, size_t());
  MOCK_METHOD1(EndEvent, void(size_t));

  /* planner */
  MOCK_METHOD0(Trigger, void());
  MOCK_METHOD2(EnqueueRequest, JobId(Job, bool));
  MOCK_METHOD2(EnqueueBatch, std::vector<JobId>(std::vector<Job>, bool));
  MOCK_METHOD1(PrepareReenqueue, void(Job&));
  MOCK_METHOD1(EnqueueFinishedJob, void(Job&));
  MOCK_METHOD1(EnqueueToWorker, bool(const ScheduleAction&));
  MOCK_METHOD1(EnqueueToWorkerBatch, bool(const std::vector<ScheduleAction>&));

  /* getters */
  ErrorReporter* GetErrorReporter() { return DefaultErrorReporter(); }
  MOCK_METHOD1(GetWorker, Worker*(WorkerId));
  MOCK_CONST_METHOD1(GetWorker, const Worker*(WorkerId));
  MOCK_CONST_METHOD0(GetNumWorkers, size_t());
  MOCK_CONST_METHOD1(GetWorkerDevice, DeviceFlag(WorkerId));

  MOCK_CONST_METHOD0(GetFrequency, Frequency*());
  MOCK_CONST_METHOD0(GetThermal, Thermal*());
  MOCK_METHOD0(SetMinFrequencies, void());
  MOCK_METHOD0(SetMaxFrequencies, void());
  MOCK_METHOD0(Sleep, void());

  /* tensor communication */
  MOCK_METHOD1(TryCopyInputTensors, absl::Status(const Job&));
  MOCK_METHOD1(TryCopyOutputTensors, absl::Status(const Job&));

  MOCK_METHOD1(BeginEvent, void(JobId));
  MOCK_METHOD1(EndEvent, void(JobId));
};

}  // namespace test
}  // namespace band

#endif  // BAND_TEST_TEST_UTIL_H_