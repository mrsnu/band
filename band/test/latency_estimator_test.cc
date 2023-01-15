#include "band/latency_estimator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/config_builder.h"
#include "band/context.h"
#include "band/cpu.h"
#include "band/test/test_util.h"
#include "band/time.h"
#include "band/worker.h"

namespace Band {
namespace Test {
struct CustomWorkerMockContext : public MockContextBase {
  CustomWorkerMockContext() { model_spec.path = "dummy"; }
  Worker* GetWorker(WorkerId id) override { return worker; }
  size_t GetNumWorkers() const override { return 1; }
  const ModelSpec* GetModelSpec(ModelId model_id) const override {
    return &model_spec;
  }
  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override {
    return SubgraphKey(0, 0);
  }
  bool HasSubgraph(const SubgraphKey& key) const override { return true; }

  Worker* worker;
  ModelSpec model_spec;
};

struct CustomInvokeMockContext : public CustomWorkerMockContext {
  CustomInvokeMockContext(
      std::function<BandStatus(const SubgraphKey&)> invoke_lambda)
      : invoke_lambda(invoke_lambda) {}

  std::function<BandStatus(const SubgraphKey&)> invoke_lambda;
  BandStatus Invoke(const Band::SubgraphKey& subgraph_key) override {
    return invoke_lambda(subgraph_key);
  }
};

using WorkerTypeList = testing::Types<DeviceQueueWorker, GlobalQueueWorker>;
template <class>
struct WorkerTypesSuite : testing::Test {};
TYPED_TEST_SUITE(WorkerTypesSuite, WorkerTypeList);

TYPED_TEST(WorkerTypesSuite, NumRunsTest) {
  CustomWorkerMockContext context;
  EXPECT_CALL(context, Invoke).Times(testing::Exactly(53));

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(50).AddNumWarmups(3).AddOnline(true).Build();

  TypeParam worker(&context, 0, kBandCPU);
  context.worker = &worker;

  worker.Start();

  LatencyEstimator latency_estimator(&context);

  EXPECT_EQ(latency_estimator.Init(config), kBandOk);
  EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);

  worker.End();
}

struct AffinityMasksFixture : public testing::TestWithParam<BandCPUMaskFlags> {
};

TEST_P(AffinityMasksFixture, AffinityPropagateTest) {
  CustomInvokeMockContext context([](const Band::SubgraphKey& subgraph_key) {
    CpuSet thread_cpu_set;
    if (GetCPUThreadAffinity(thread_cpu_set) != kBandOk) {
      return kBandError;
    }
    // TODO(BAND-20): propagate affinity to CPU backend, and set number of
    // threads
    CpuSet target_set = BandCPUMaskGetSet(GetParam());
    if (target_set.NumEnabled() == 0) {
      return kBandOk;
    } else {
      return thread_cpu_set == BandCPUMaskGetSet(GetParam()) ? kBandOk
                                                             : kBandError;
    }
  });

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(3).AddNumWarmups(3).AddOnline(true).Build();

  DeviceQueueWorker worker(&context, 0, kBandCPU);
  // Explicitly assign worker to mock context
  context.worker = &worker;
  // Update worker thread affinity
  worker.UpdateWorkerThread(BandCPUMaskGetSet(GetParam()), 3);
  worker.Start();

  LatencyEstimator latency_estimator(&context);

  EXPECT_EQ(latency_estimator.Init(config), kBandOk);
  // This will be kBandError if affinity propagation fails
  EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);

  worker.End();
}

INSTANTIATE_TEST_SUITE_P(AffinityPropagateTests, AffinityMasksFixture,
                         testing::Values(kBandAll, kBandLittle, kBandBig,
                                         kBandPrimary));

TEST(LatencyEstimatorSuite, OnlineLatencyProfile) {
  CustomInvokeMockContext context([](const Band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return kBandOk;
  });

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(3).AddNumWarmups(3).AddOnline(true).Build();

  DeviceQueueWorker worker(&context, 0, kBandCPU);
  // Explicitly assign worker to mock context
  context.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  LatencyEstimator latency_estimator(&context);

  EXPECT_EQ(latency_estimator.Init(config), kBandOk);
  EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
  EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);
  EXPECT_GT(latency_estimator.GetProfiled(key), 5000);

  worker.End();
}

TEST(LatencyEstimatorSuite, OfflineSaveLoadSuccess) {
  CustomInvokeMockContext context([](const Band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return kBandOk;
  });

  DeviceQueueWorker worker(&context, 0, kBandCPU);
  // explicitly assign worker to mock context
  context.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  {
    // profile on online estimator
    LatencyEstimator latency_estimator(&context);

    ProfileConfigBuilder b;
    ProfileConfig config =
        b.AddNumRuns(3)
            .AddNumWarmups(3)
            .AddOnline(true)
            .AddProfileDataPath(testing::TempDir() + "log.json")
            .Build();

    EXPECT_EQ(latency_estimator.Init(config), kBandOk);
    EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);
    EXPECT_EQ(latency_estimator.DumpProfile(), kBandOk);
  }

  {
    // load on offline estimator
    LatencyEstimator latency_estimator(&context);

    ProfileConfigBuilder b;
    ProfileConfig config =
        b.AddNumRuns(3)
            .AddNumWarmups(3)
            .AddOnline(false)
            .AddProfileDataPath(testing::TempDir() + "log.json")
            .Build();

    EXPECT_EQ(latency_estimator.Init(config), kBandOk);
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
    EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);
    EXPECT_GT(latency_estimator.GetProfiled(key), 5000);
  }

  worker.End();
}

TEST(LatencyEstimatorSuite, OfflineSaveLoadFailure) {
  CustomInvokeMockContext context([](const Band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return kBandOk;
  });

  DeviceQueueWorker worker(&context, 0, kBandCPU);
  // explicitly assign worker to mock context
  context.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  {
    // profile on online estimator
    LatencyEstimator latency_estimator(&context);

    ProfileConfigBuilder b;
    ProfileConfig config =
        b.AddNumRuns(3)
            .AddNumWarmups(3)
            .AddOnline(true)
            .AddProfileDataPath(testing::TempDir() + "log.json")
            .Build();

    EXPECT_EQ(latency_estimator.Init(config), kBandOk);
    EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);
    EXPECT_EQ(latency_estimator.DumpProfile(), kBandOk);
  }

  {
    worker.UpdateWorkerThread(worker.GetWorkerThreadAffinity(),
                              worker.GetNumThreads() + 1);
    // load on offline estimator
    LatencyEstimator latency_estimator(&context);

    ProfileConfigBuilder b;
    ProfileConfig config =
        b.AddNumRuns(3)
            .AddNumWarmups(3)
            .AddOnline(false)
            .AddProfileDataPath(testing::TempDir() + "log.json")
            .Build();

    EXPECT_EQ(latency_estimator.Init(config), kBandOk);
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
    EXPECT_EQ(latency_estimator.ProfileModel(0), kBandOk);
    // fails to load due to worker update
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
  }

  worker.End();
}

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}