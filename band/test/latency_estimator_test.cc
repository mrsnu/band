#include "band/latency_estimator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"
#include "band/cpu.h"
#include "band/time.h"
#include "band/worker.h"

namespace Band {
namespace Test {

struct MockContext : public Context {
  MOCK_METHOD(std::vector<JobId>, EnqueueBatch, (std::vector<Job>, bool),
              (override));

  void PrepareReenqueue(Job&) override{};
  void UpdateLatency(const SubgraphKey&, int64_t) override{};
  void EnqueueFinishedJob(Job& job) override {}
  void Trigger() override {}
  MOCK_METHOD(BandStatus, Invoke, (const SubgraphKey&), (override));

  Worker* GetWorker(WorkerId id) override { return worker; }
  size_t GetNumWorkers() const { return 1; }

  Worker* worker;
};

using WorkerTypeList = testing::Types<DeviceQueueWorker, GlobalQueueWorker>;
template <class>
struct WorkerTypesSuite : testing::Test {};
TYPED_TEST_SUITE(WorkerTypesSuite, WorkerTypeList);

TYPED_TEST(WorkerTypesSuite, NumRunsTest) {
  MockContext context;
  EXPECT_CALL(context, Invoke).Times(testing::Exactly(53));

  TypeParam worker(&context, kBandCPU);
  context.worker = &worker;

  worker.Start();

  LatencyEstimator latency_estimator(&context);
  ProfileConfig config;
  config.num_runs = 50;
  config.num_warmups = 3;
  config.online = true;

  EXPECT_EQ(latency_estimator.Init(config), kBandOk);
  EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);

  worker.End();
}

struct AffinityMasksFixture : public testing::TestWithParam<BandCPUMaskFlags> {
};

TEST_P(AffinityMasksFixture, AffinityPropagateTest) {
  MockContext context;

  ON_CALL(context, Invoke)
      .WillByDefault(testing::Invoke([](const Band::SubgraphKey& subgraph_key) {
        CpuSet thread_cpu_set;
        if (GetCPUThreadAffinity(thread_cpu_set) != kBandOk) {
          return kBandError;
        }
        // TODO(BAND-20): propagate affinity to CPU backend, and set number of
        // threads
        return thread_cpu_set == BandCPUMaskGetSet(GetParam()) ? kBandOk
                                                               : kBandError;
      }));

  EXPECT_CALL(context, Invoke).WillRepeatedly(testing::Return(kBandOk));

  DeviceQueueWorker worker(&context, kBandCPU);
  // Explicitly assign worker to mock context
  context.worker = &worker;
  // Update worker thread affinity
  worker.UpdateWorkerThread(BandCPUMaskGetSet(GetParam()), 3);
  worker.Start();

  LatencyEstimator latency_estimator(&context);
  ProfileConfig config;
  config.num_runs = 3;
  config.num_warmups = 3;
  config.online = true;

  EXPECT_EQ(latency_estimator.Init(config), kBandOk);
  EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);

  worker.End();
}

INSTANTIATE_TEST_SUITE_P(AffinityPropagateTests, AffinityMasksFixture,
                         testing::Values(kBandAll, kBandLittle, kBandBig,
                                         kBandPrimary));

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}