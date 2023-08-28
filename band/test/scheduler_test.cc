#include "band/scheduler/scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/config.h"
#include "band/model.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/scheduler/least_slack_first_scheduler.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/test/test_util.h"

namespace band {
namespace test {

struct MockEngine : public MockEngineBase {
  std::set<WorkerId> idle_workers_;
  std::vector<WorkerId> list_idle_workers_;
  std::map<ModelId, WorkerId> model_worker_map_;
  std::vector<ScheduleAction> action_;
  mutable int w;

  MockEngine(std::set<WorkerId> idle_workers) : idle_workers_(idle_workers) {
    w = 0;
    list_idle_workers_.assign(idle_workers.begin(), idle_workers.end());
  }

  std::set<WorkerId> GetIdleWorkers() const override { return idle_workers_; }

  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override {
    return SubgraphKey(model_id, worker_id, {0});
  }

  std::pair<std::vector<SubgraphKey>, double> GetSubgraphWithMinCost(
      const Job& job, const WorkerWaitingTime& worker_waiting,
      std::function<double(double, std::map<SensorFlag, double>, std::map<SensorFlag, double>)>)
      const override {
    return std::pair<std::vector<SubgraphKey>, double>(
        {SubgraphKey(job.model_id, *idle_workers_.begin(), {0})},
        0 /*shortest expected latency*/);
  }

  WorkerId GetModelWorker(ModelId model_id) const override {
    if (w > list_idle_workers_.size())
      return -1;
    else
      return list_idle_workers_[w++];
  }

  WorkerWaitingTime GetWorkerWaitingTime() const override {
    WorkerWaitingTime map;
    for (auto worker_id : list_idle_workers_) {
      map[worker_id] = 0;
    }
    return map;
  }

  int64_t GetExpected(const SubgraphKey& key) const override { return 10; }
  bool EnqueueToWorker(const ScheduleAction& action) override {
    action_.push_back(action);
    return true;
  }

  bool EnqueueToWorkerBatch(
      const std::vector<ScheduleAction>& schedule_action) override {
    action_.insert(action_.end(), schedule_action.begin(),
                   schedule_action.end());
    return true;
  }
};

// <request model ids, request slos, available workers>
struct LSTTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::deque<int>, std::set<int>>> {};

// <request model ids, available workers>
struct ModelLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::set<int>>> {};

// <request config path string, available workers>
struct ConfigLevelTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int>, std::set<int>>> {};

TEST_P(LSTTestsFixture, LSTTest) {
  std::deque<int> request_models = std::get<0>(GetParam());
  std::deque<int> request_slos = std::get<1>(GetParam());
  std::set<int> available_workers = std::get<2>(GetParam());

  assert(request_models.size() == request_slos.size());

  std::deque<Job> requests;
  for (int i = 0; i < request_models.size(); i++) {
    requests.emplace_back(Job(request_models[i], request_slos[i]));
  }
  const int count_requests = requests.size();

  MockEngine engine(available_workers);
  LeastSlackFirstScheduler lst_scheduler(engine, 5);
  lst_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_models : engine.action_) {
    count_scheduled++;
  }

  EXPECT_EQ(count_scheduled,
            std::min(available_workers.size(), request_models.size()));
  EXPECT_EQ(count_requests, requests.size() + count_scheduled);
  if (request_slos[0] == 0) {  // No SLOs
    EXPECT_EQ(engine.action_[0].second.GetModelId(), 0);
    EXPECT_EQ(engine.action_[1].second.GetModelId(), 1);
  } else {  // SLOs
    EXPECT_EQ(engine.action_[0].second.GetModelId(), 1);
    EXPECT_EQ(engine.action_[1].second.GetModelId(), 0);
  }
}

TEST_P(ModelLevelTestsFixture, RoundRobinTest) {
  std::deque<int> request_models = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());

  std::deque<Job> requests;
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    requests.emplace_back(Job(*it));
  }
  const int count_requests = requests.size();

  MockEngine engine(available_workers);
  RoundRobinScheduler rr_scheduler(engine);
  rr_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_models : engine.action_) {
    count_scheduled++;
  }

  EXPECT_EQ(count_scheduled,
            std::min(available_workers.size(), request_models.size()));
  EXPECT_EQ(count_requests, requests.size() + count_scheduled);
}

TEST_P(ConfigLevelTestsFixture, FixedDeviceFixedWorkerTest) {
  // Set configs in engine
  std::deque<int> request_models = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());

  std::deque<Job> requests;
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    requests.emplace_back(Job(*it));
  }
  const int count_requests = requests.size();

  MockEngine engine(available_workers);
  FixedWorkerScheduler fd_scheduler(engine);
  fd_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_models : engine.action_) {
    count_scheduled++;
  }

  // Each model made a single request and should be scheduled once
  EXPECT_EQ(count_scheduled, count_requests);
  // requests should be deleted
  EXPECT_EQ(requests.size(), 0);

  std::map<ModelId, int> scheduled_models;
  // each worker should have a single model scheduled
  for (auto action : engine.action_) {
    scheduled_models[action.second.GetModelId()]++;
    EXPECT_EQ(scheduled_models[action.second.GetModelId()], 1);
  }
  // Each requested model should be scheduled
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    EXPECT_NE(scheduled_models.find((*it)), scheduled_models.end());
  }
}

TEST_P(ConfigLevelTestsFixture, FixedDeviceFixedWorkerEngineRequestTest) {
  // Set configs in engine
  std::deque<int> request_models = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());
  const int target_worker = 0;

  std::deque<Job> requests;
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    Job job = Job(*it);
    job.target_worker_id = target_worker;
    requests.emplace_back(job);
  }
  const int count_requests = requests.size();

  MockEngine engine(available_workers);
  FixedWorkerScheduler fd_scheduler(engine);
  fd_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_jobs : engine.action_) {
    count_scheduled++;
  }

  // Each model made a single request and should be scheduled once
  EXPECT_EQ(count_scheduled, count_requests);
  // requests should be deleted
  EXPECT_EQ(requests.size(), 0);

  std::map<ModelId, int> scheduled_models;
  // each worker should have a single model scheduled
  EXPECT_EQ(engine.action_.size(), count_requests);
  for (auto scheduled_model : engine.action_) {
    for (auto action : engine.action_) {
      scheduled_models[action.second.GetModelId()]++;
    }
  }
  // Each requested model should be scheduled
  for (auto it = request_models.begin(); it != request_models.end(); it++) {
    EXPECT_NE(scheduled_models.find((*it)), scheduled_models.end());
  }
}

INSTANTIATE_TEST_SUITE_P(
    LSTTests, LSTTestsFixture,
    testing::Values(
        std::make_tuple(std::deque<int>{0, 1}, std::deque<int>{100, 80},
                        std::set<int>{0, 1, 2}),  // With SLO
        std::make_tuple(std::deque<int>{0, 1}, std::deque<int>{0, 0},
                        std::set<int>{0, 1, 2})  // Without SLO
        ));

INSTANTIATE_TEST_SUITE_P(
    RoundRobinTests, ModelLevelTestsFixture,
    testing::Values(
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1}, std::set<int>{0, 1, 2}),
        std::make_tuple(std::deque<int>{0, 1, 2}, std::set<int>{0, 1})));

INSTANTIATE_TEST_SUITE_P(
    FixedDeviceFixedWorkerTests, ConfigLevelTestsFixture,
    testing::Values(std::make_tuple(std::deque<int>{0, 1, 2},
                                    std::set<int>{0, 1, 2})));
INSTANTIATE_TEST_SUITE_P(
    FixedDeviceFixedWorkerEngineRequestTests, ConfigLevelTestsFixture,
    testing::Values(std::make_tuple(std::deque<int>{0, 1, 2},
                                    std::set<int>{0, 1, 2})));

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}