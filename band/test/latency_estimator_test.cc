#include "band/latency_estimator.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/config_builder.h"
#include "band/device/cpu.h"
#include "band/engine_interface.h"
#include "band/model_spec.h"
#include "band/test/test_util.h"
#include "band/time.h"
#include "band/worker.h"

namespace band {
namespace test {
struct CustomWorkerMockContext : public MockContextBase {
  CustomWorkerMockContext() { model_spec.path = "dummy"; }
  Worker* GetWorker(WorkerId id) override { return worker; }
  size_t GetNumWorkers() const override { return 1; }
  const ModelSpec* GetModelSpec(ModelId model_id) const override {
    return &model_spec;
  }
  void ForEachSubgraph(
      std::function<void(const SubgraphKey&)> iterator) const override {
    iterator(SubgraphKey(0, 0));
  }
  bool HasSubgraph(const SubgraphKey& key) const override { return true; }

  Worker* worker;
  ModelSpec model_spec;
};

struct CustomInvokeMockContext : public CustomWorkerMockContext {
  CustomInvokeMockContext(
      std::function<absl::Status(const SubgraphKey&)> invoke_lambda)
      : invoke_lambda(invoke_lambda) {}

  std::function<absl::Status(const SubgraphKey&)> invoke_lambda;
  absl::Status Invoke(const band::SubgraphKey& subgraph_key) override {
    return invoke_lambda(subgraph_key);
  }
};

using WorkerTypeList = testing::Types<DeviceQueueWorker, GlobalQueueWorker>;
template <class>
struct WorkerTypesSuite : testing::Test {};
TYPED_TEST_SUITE(WorkerTypesSuite, WorkerTypeList);

TYPED_TEST(WorkerTypesSuite, NumRunsTest) {
  CustomWorkerMockContext engine;
  EXPECT_CALL(engine, Invoke).Times(testing::Exactly(53));

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(50).AddNumWarmups(3).AddOnline(true).Build();

  TypeParam worker(&engine, 0, DeviceFlag::kCPU);
  engine.worker = &worker;

  worker.Start();

  LatencyEstimator latency_estimator(&engine);

  EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
  EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());

  worker.End();
}

struct CPUMaskFixture : public testing::TestWithParam<CPUMaskFlag> {};

TEST_P(CPUMaskFixture, AffinityPropagateTest) {
  CustomInvokeMockContext engine([](const band::SubgraphKey& subgraph_key) {
    CpuSet thread_cpu_set;
    if (!GetCPUThreadAffinity(thread_cpu_set).ok()) {
      return absl::InternalError("GetCPUThreadAffinity");
    }
    // TODO(#238): propagate affinity to CPU backend, and set number of
    // threads
    CpuSet target_set = BandCPUMaskGetSet(GetParam());
    if (target_set.NumEnabled() == 0) {
      return absl::OkStatus();
    } else {
      return thread_cpu_set == BandCPUMaskGetSet(GetParam())
                 ? absl::OkStatus()
                 : absl::InternalError("");
    }
  });

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(3).AddNumWarmups(3).AddOnline(true).Build();

  DeviceQueueWorker worker(&engine, 0, DeviceFlag::kCPU);
  // Explicitly assign worker to mock engine
  engine.worker = &worker;
  // Update worker thread affinity
  auto status = worker.UpdateWorkerThread(BandCPUMaskGetSet(GetParam()), 3);
  EXPECT_EQ(status, absl::OkStatus());
  worker.Start();

  LatencyEstimator latency_estimator(&engine);

  EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
  // This will be kBandError if affinity propagation fails
  EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());

  worker.End();
}

INSTANTIATE_TEST_SUITE_P(AffinityPropagateTests, CPUMaskFixture,
                         testing::Values(CPUMaskFlag::kAll,
                                         CPUMaskFlag::kLittle,
                                         CPUMaskFlag::kBig,
                                         CPUMaskFlag::kPrimary));

TEST(LatencyEstimatorSuite, OnlineLatencyProfile) {
  CustomInvokeMockContext engine([](const band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return absl::OkStatus();
  });

  ProfileConfigBuilder b;
  ProfileConfig config =
      b.AddNumRuns(3).AddNumWarmups(3).AddOnline(true).Build();

  DeviceQueueWorker worker(&engine, 0, DeviceFlag::kCPU);
  // Explicitly assign worker to mock engine
  engine.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  LatencyEstimator latency_estimator(&engine);

  EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
  EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
  EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());
  EXPECT_GT(latency_estimator.GetProfiled(key), 5000);

  worker.End();
}

TEST(LatencyEstimatorSuite, OfflineSaveLoadSuccess) {
  CustomInvokeMockContext engine([](const band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return absl::OkStatus();
  });

  const std::string profile_path = testing::TempDir() + "log.json";

  DeviceQueueWorker worker(&engine, 0, DeviceFlag::kCPU);
  // explicitly assign worker to mock engine
  engine.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  {
    // profile on online estimator
    LatencyEstimator latency_estimator(&engine);

    ProfileConfigBuilder b;
    ProfileConfig config = b.AddNumRuns(3)
                               .AddNumWarmups(3)
                               .AddOnline(true)
                               .AddProfileDataPath(profile_path)
                               .Build();

    EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
    EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());
    EXPECT_EQ(latency_estimator.DumpProfile(), absl::OkStatus());
  }

  {
    // load on offline estimator
    LatencyEstimator latency_estimator(&engine);

    ProfileConfigBuilder b;
    ProfileConfig config = b.AddNumRuns(3)
                               .AddNumWarmups(3)
                               .AddOnline(false)
                               .AddProfileDataPath(profile_path)
                               .Build();

    EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
    EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());
    EXPECT_GT(latency_estimator.GetProfiled(key), 5000);
  }

  std::remove(profile_path.c_str());

  worker.End();
}

TEST(LatencyEstimatorSuite, OfflineSaveLoadFailure) {
  CustomInvokeMockContext engine([](const band::SubgraphKey& subgraph_key) {
    std::this_thread::sleep_for(std::chrono::microseconds(5000));
    return absl::OkStatus();
  });

  const std::string profile_path = testing::TempDir() + "log.json";

  DeviceQueueWorker worker(&engine, 0, DeviceFlag::kCPU);
  // explicitly assign worker to mock engine
  engine.worker = &worker;
  worker.Start();
  SubgraphKey key(0, 0);

  {
    // profile on online estimator
    LatencyEstimator latency_estimator(&engine);

    ProfileConfigBuilder b;
    ProfileConfig config = b.AddNumRuns(3)
                               .AddNumWarmups(3)
                               .AddOnline(true)
                               .AddProfileDataPath(profile_path)
                               .Build();

    EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
    EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());
    EXPECT_EQ(latency_estimator.DumpProfile(), absl::OkStatus());
  }

  {
    auto status = worker.UpdateWorkerThread(worker.GetWorkerThreadAffinity(),
                                            worker.GetNumThreads() + 1);
    EXPECT_EQ(status, absl::OkStatus());
    // load on offline estimator
    LatencyEstimator latency_estimator(&engine);

    ProfileConfigBuilder b;
    ProfileConfig config = b.AddNumRuns(3)
                               .AddNumWarmups(3)
                               .AddOnline(false)
                               .AddProfileDataPath(profile_path)
                               .Build();

    EXPECT_EQ(latency_estimator.Init(config), absl::OkStatus());
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
    EXPECT_EQ(latency_estimator.ProfileModel(0), absl::OkStatus());
    // fails to load due to worker update
    EXPECT_EQ(latency_estimator.GetProfiled(key), -1);
  }

  std::remove(profile_path.c_str());

  worker.End();
}

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}