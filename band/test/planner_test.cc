#include "band/planner.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/scheduler/scheduler.h"
#include "band/test/test_util.h"
#include "band/time.h"
#include "band/worker.h"

namespace band {
namespace test {

struct MockContext : public MockContextBase {
  void PrepareReenqueue(Job&) override{};
  void UpdateLatency(const SubgraphKey&, int64_t) override{};
  void EnqueueFinishedJob(Job& job) override { finished.insert(job.job_id); }
  void Trigger() override {}

  absl::Status Invoke(const SubgraphKey& key) override {
    Time::SleepForMicros(50);
    return absl::OkStatus();
  }

  std::set<int> finished;
};

class MockScheduler : public IScheduler {
  using IScheduler::IScheduler;

  MOCK_METHOD1(Schedule, void(JobQueue&));
  MOCK_METHOD0(NeedProfile, bool());
  MOCK_METHOD0(NeedFallbackSubgraphs, bool());
  // MOCK_METHOD0(GetWorkerType, WorkerType());
  WorkerType GetWorkerType() { return WorkerType::DeviceQueue; }
};

/*

Job cycle

planner -> scheduler -> worker -> planner

*/

TEST(PlannerSuite, SingleQueue) {
  MockContext context;
  Planner planner(context);
  auto status = planner.AddScheduler(std::make_unique<MockScheduler>(context));
  EXPECT_EQ(status, absl::OkStatus());
  // TODO: Add tests!
  EXPECT_TRUE(true);
}
}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}