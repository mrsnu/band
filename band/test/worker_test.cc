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

#include "band/worker.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <future>

#include "band/test/test_util.h"
#include "band/time.h"

namespace band {
namespace test {

struct MockEngine : public MockEngineBase {
  void EnqueueFinishedJob(Job& job) override { finished.insert(job.job_id); }
  absl::Status Invoke(const SubgraphKey& key) override {
    time::SleepForMicros(50);
    return absl::OkStatus();
  }
  std::set<int> finished;
};

using WorkerTypeList = testing::Types<DeviceQueueWorker, GlobalQueueWorker>;
template <class>
struct WorkerSuite : testing::Test {};
TYPED_TEST_SUITE(WorkerSuite, WorkerTypeList);

Job GetEmptyJob() {
  Job job(0);
  job.subgraph_key = SubgraphKey(0, 0);
  job.enqueue_time = time::NowMicros();
  return job;
}

TYPED_TEST(WorkerSuite, JobHelper) {
  MockEngine engine;
  TypeParam worker(&engine, 0, DeviceFlag::kCPU);
  Job job = GetEmptyJob();

  worker.Start();

  EXPECT_FALSE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), -1);

  EXPECT_TRUE(worker.EnqueueJob(job));
  EXPECT_TRUE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), job.job_id);

  worker.End();
}

TYPED_TEST(WorkerSuite, Wait) {
  MockEngine engine;
  EXPECT_CALL(engine, UpdateLatency).Times(testing::AtLeast(1));
  EXPECT_CALL(engine, Trigger).Times(testing::AtLeast(1));
  EXPECT_CALL(engine, TryCopyInputTensors).Times(testing::AtLeast(1));
  EXPECT_CALL(engine, TryCopyOutputTensors).Times(testing::AtLeast(1));

  TypeParam worker(&engine, 0, DeviceFlag::kCPU);
  Job job = GetEmptyJob();

  worker.Start();

  EXPECT_FALSE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), -1);

  auto now0 = time::NowMicros();
  EXPECT_TRUE(worker.EnqueueJob(job));
  worker.Wait();
  auto now1 = time::NowMicros();
  EXPECT_GE(now1, now0 + 50);

  EXPECT_NE(engine.finished.find(job.job_id), engine.finished.end());
  worker.End();
}

// TODO: throttling test
}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}