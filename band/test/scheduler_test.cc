#include "band/scheduler/scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/context.h"
#include "band/config.h"
#include "band/model.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/scheduler/fixed_device_scheduler.h"

namespace Band {
namespace Test {

struct MockContext : public Context {
  std::set<WorkerId> idle_workers_;
  std::vector<WorkerId> list_idle_workers_;
  std::map<ModelId, WorkerId> model_worker_map_;
  mutable int w;
  
  MockContext(std::set<WorkerId> idle_workers) : idle_workers_(idle_workers) {
    w = 0;
    list_idle_workers_.assign(idle_workers.begin(), idle_workers.end());
  }

  std::set<WorkerId> GetIdleWorkers() const override { return idle_workers_; }

  SubgraphKey GetModelSubgraphKey(ModelId model_id,
                                  WorkerId worker_id) const override {
    return SubgraphKey(model_id, worker_id, {0}, {0});
  }

  WorkerId GetModelWorker(ModelId model_id) const override{
    if(w > list_idle_workers_.size()) return -1;
    else return list_idle_workers_[w++];
  }

};

// <request model ids, available workers>
struct ModelLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::set<int>>> {};

// <request config path string, available workers>
struct ConfigLevelTestsFixture
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

TEST_P(ConfigLevelTestsFixture, FixedDeviceTest){
  // Set configs in context
  std::deque<int> request_models = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());  

  std::deque<Job> requests;
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    requests.emplace_back(Job(*it));
  }
  const int count_requests = requests.size();

  MockContext context(available_workers);
  FixedDeviceScheduler fd_scheduler;
  auto action = fd_scheduler.Schedule(context, requests);
  
  int count_scheduled = 0;
  for(auto scheduled_jobs: action){
    count_scheduled += scheduled_jobs.second.size();
  }
  
  // Each model made a single request and should be scheduled once
  EXPECT_EQ(count_scheduled, count_requests);
  // requests should be deleted
  EXPECT_EQ(requests.size(), 0);
  
  
  std::set<ModelId> scheduled_models;
  // each worker should have a single model scheduled
  for(auto scheduled_model: action){
    EXPECT_EQ(scheduled_model.second.size(), 1);
    if(scheduled_model.second.size() == 1){
      scheduled_models.insert(scheduled_model.second.at(0).first.model_id);
    }
  }
  // Each requested model should be scheduled
  for(auto it = request_models.begin(); it != request_models.end(); it++){
    EXPECT_NE(scheduled_models.find((*it)), scheduled_models.end());
  }
}

INSTANTIATE_TEST_SUITE_P(
    RoundRobinTests, ModelLevelTestsFixture,
    testing::Values(
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1})));

INSTANTIATE_TEST_SUITE_P(
  FixedDeviceTests, ConfigLevelTestsFixture,
  testing::Values(
    std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1, 2})
  )
);

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}