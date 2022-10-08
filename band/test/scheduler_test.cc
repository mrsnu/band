#include "band/scheduler/scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"
#include "band/scheduler/round_robin_scheduler.h"

namespace Band {
namespace Test {

struct MockContext : public Context {
  MockContext(std::set<WorkerId> idle_workers) : idle_workers_(idle_workers) {}

  std::set<WorkerId> GetIdleWorkers() const override { return idle_workers_; }

  // absl::StatusOr<SubgraphKey> GetModelSubgraphKey(ModelId model_id,
  //                                 WorkerId worker_id) const override {
  //   return SubgraphKey(model_id, worker_id, {0}, {0});
  // }

  std::set<WorkerId> idle_workers_;
};

// <request model ids, available workers>
struct ModelLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::set<int>>> {};

TEST_P(ModelLevelTestsFixture, RoundRobinTest) {
  std::deque<int> request_models = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());

  std::deque<Job> requests;
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    requests.emplace_back(Job(*it));
  }
  const int count_requests = requests.size();

  MockContext context(available_workers);
  RoundRobinScheduler rr_scheduler;
  auto action = rr_scheduler.Schedule(context, requests);

  int count_scheduled = 0;
  for (auto scheduled_models : action) {
    count_scheduled += scheduled_models.second.size();
  }

  EXPECT_EQ(count_scheduled,
            std::min(available_workers.size(), request_models.size()));
  EXPECT_EQ(count_requests, requests.size() + count_scheduled);
}

INSTANTIATE_TEST_SUITE_P(
    RoundRobinTests, ModelLevelTestsFixture,
    testing::Values(
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1})));
}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}