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
  std::map<ModelId, ModelConfig> model_configs_;
  std::map<ModelId, WorkerId> model_worker_map_;
  std::deque<Job> requests;
  
  MockContext(std::set<WorkerId> idle_workers) : idle_workers_(idle_workers) {}

  std::set<WorkerId> GetIdleWorkers() const override { return idle_workers_; }

  SubgraphKey GetModelSubgraphKey(ModelId model_id,
                                  WorkerId worker_id) const override {
    return SubgraphKey(model_id, worker_id, {0}, {0});
  }

  BandStatus Init(const RuntimeConfig& config) override {
    WorkerId worker_id = *idle_workers_.begin();
    for(auto model_config: config.interpreter_config.models_config){
      Model model;
      ModelId model_id;
      if(model.FromPath(kBandTfLite, model_config.model_fname.c_str()) != kBandOk){
        error_reporter_->Report("Model %s could not be instantiated for %s.",
        model_config.model_fname, BandBackendGetName(kBandTfLite));
      }
      model_id = model.GetId();
      model_configs_[model_id] = model_config;
      model_worker_map_[model_id] = worker_id++;
      requests.emplace_back(Job(model_id));
    }
    return kBandOk;
  }

  const ModelConfig* GetModelConfig(ModelId model_id) const override { return &(model_configs_.at(model_id));}

  WorkerId GetModelWorker(ModelId model_id) const override{
   return model_worker_map_.at(model_id);
  }
};

// <request model ids, available workers>
struct ModelLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::set<int>>> {};

// <request config path string, available workers>
struct ConfigLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::string, std::set<int>>> {};

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
  std::string request_config_string = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());

  RuntimeConfig request_config;
  ParseRuntimeConfigFromJson(request_config_string, request_config);

  MockContext context(available_workers);
  context.Init(request_config);
  FixedDeviceScheduler fd_scheduler;
  auto requests = context.requests;
  auto action = fd_scheduler.Schedule(context, context.requests);
  
  int count_scheduled = 0;
  for(auto scheduled_jobs: action){
    count_scheduled += scheduled_jobs.second.size();
  }
  
  // Each model should be scheduled
  EXPECT_EQ(count_scheduled, requests.size());
  // requests should be deleted
  EXPECT_EQ(context.requests.size(), 0);
  
  std::set<ModelId> requested_models;
  std::set<ModelId> scheduled_models;
  for(auto it = requests.begin(); it != requests.end(); it++){
    requested_models.insert((*it).model_id);
  }
  // each worker should have a single model scheduled
  for(auto scheduled_model: action){
    EXPECT_EQ(scheduled_model.second.size(), 1);
    if(scheduled_model.second.size() == 1){
      scheduled_models.insert(scheduled_model.second.at(0).first.model_id);
    }
  }
  // Each requested model should be scheduled
  for(auto it = requested_models.begin(); it != requested_models.end(); it++){
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
    std::make_tuple("band/testdata/config.json", std::set<int>{0, 1, 2})
  )
);

}  // namespace Test
}  // namespace Band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}