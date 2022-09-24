#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"
#include "band/time.h"
#include "band/worker.h"

namespace Band {
namespace Test {

struct MockContext : public Context {
  MOCK_METHOD(std::vector<JobId>, EnqueueBatch, (std::vector<Job>, bool),
              (override));

  MOCK_METHOD(void, UpdateLatency, (const SubgraphKey &, int64_t), (override));
  void EnqueueFinishedJob(Job &job) override { finished.insert(job.job_id); }
  MOCK_METHOD(void, Trigger, (), (override));

  BandStatus Invoke(const SubgraphKey &key) override {
    Time::SleepForMicros(50);
    return kBandOk;
  }

  std::set<int> finished;
};

using WorkerTypeList = testing::Types<DeviceQueueWorker, GlobalQueueWorker>;
template <class> struct WokrerSuite : testing::Test {};
TYPED_TEST_SUITE(WokrerSuite, WorkerTypeList);

Job GetEmptyJob() {
  Job job(0);
  job.subgraph_key = SubgraphKey(0, 0);
  job.enqueue_time = Time::NowMicros();
  return job;
}

TYPED_TEST(WokrerSuite, JobHelper) {
  MockContext context;
  TypeParam worker(&context, kBandCPU);
  Job job = GetEmptyJob();

  worker.Start();

  EXPECT_FALSE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), -1);

  EXPECT_TRUE(worker.GiveJob(job));
  EXPECT_TRUE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), job.job_id);

  worker.End();
}

TYPED_TEST(WokrerSuite, Wait) {
  MockContext context;
  EXPECT_CALL(context, UpdateLatency).Times(testing::AtLeast(1));
  EXPECT_CALL(context, Trigger).Times(testing::AtLeast(1));

  TypeParam worker(&context, kBandCPU);
  Job job = GetEmptyJob();

  worker.Start();

  EXPECT_FALSE(worker.HasJob());
  EXPECT_EQ(worker.GetCurrentJobId(), -1);

  auto now0 = Time::NowMicros();
  EXPECT_TRUE(worker.GiveJob(job));
  worker.Wait();
  auto now1 = Time::NowMicros();
  EXPECT_GE(now1, now0 + 50);

  EXPECT_NE(context.finished.find(job.job_id), context.finished.end());
  worker.End();
}

// TODO: throttling test
} // namespace Test
} // namespace Band

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}