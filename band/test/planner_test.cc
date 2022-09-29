#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"
#include "band/planner.h"
#include "band/scheduler/scheduler.h"
#include "band/time.h"
#include "band/worker.h"

namespace Band {
namespace Test {

struct MockContext : public Context {
  MOCK_METHOD(std::vector<JobId>, EnqueueBatch, (std::vector<Job>, bool),
              (override));

  void PrepareReenqueue(Job &) override{};
  void UpdateLatency(const SubgraphKey &, int64_t) override{};
  void EnqueueFinishedJob(Job &job) override { finished.insert(job.job_id); }
  void Trigger() override {}

  BandStatus Invoke(const SubgraphKey &key) override {
    Time::SleepForMicros(50);
    return kBandOk;
  }

  std::set<int> finished;
};

class MockScheduler : public IScheduler {

  MOCK_METHOD(ScheduleAction, Schedule, (const Context &, JobQueue &),
              (override));
  MOCK_METHOD(bool, NeedProfile, (), (override));
  MOCK_METHOD(bool, NeedFallbackSubgraphs, (), (override));
  MOCK_METHOD(BandWorkerType, GetWorkerType, (), (override));
};
/*

Job cycle

planner -> scheduler -> worker -> planner

*/

TEST(PlannerSuite, SingleQueue) {
  MockContext context;
  Planner planner(&context);

  planner.AddScheduler(std::make_unique<MockScheduler>());
  // TODO: Add tests!
}
} // namespace Test
} // namespace Band

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}