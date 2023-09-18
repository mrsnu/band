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

#include "band/scheduler/scheduler.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "band/config.h"
#include "band/model.h"
#include "band/scheduler/fixed_worker_scheduler.h"
#include "band/scheduler/least_slack_first_scheduler.h"
#include "band/scheduler/round_robin_scheduler.h"
#include "band/scheduler/shortest_expected_latency_scheduler.h"
#include "band/scheduler/heterogeneous_earliest_finish_time_scheduler.h"
#include "band/test/test_util.h"

namespace band {
namespace test {

struct MockEngine : public MockEngineBase {
  std::set<WorkerId> idle_workers_;
  std::vector<WorkerId> list_idle_workers_;
  std::map<ModelId, WorkerId> model_worker_map_;
  std::vector<ScheduleAction> action_;
  mutable int w;
  mutable WorkerWaitingTime map;

  MockEngine(std::set<WorkerId> idle_workers) : idle_workers_(idle_workers) {
    w = 0;
    list_idle_workers_.assign(idle_workers.begin(), idle_workers.end());
    for (auto worker_id : list_idle_workers_) {
      map[worker_id] = 0;
    }
  }

  std::set<WorkerId> GetIdleWorkers() const override {
    std::set<int> idle_workers;
    for (auto worker_waiting : map) {
      if (worker_waiting.second == 0) {
        idle_workers.insert(worker_waiting.first);
      }
    }

    return idle_workers;
  }

  SubgraphKey GetLargestSubgraphKey(ModelId model_id,
                                    WorkerId worker_id) const override {
    return SubgraphKey(model_id, worker_id, {0});
  }

  std::pair<std::vector<SubgraphKey>, int64_t> GetSubgraphWithShortestLatency(
      const Job& job, const WorkerWaitingTime& worker_waiting) const override {
        auto target_worker_id = job.target_worker_id != -1 ? job.target_worker_id : *idle_workers_.begin();
    return std::pair<std::vector<SubgraphKey>, int64_t>(
        {SubgraphKey(job.model_id, target_worker_id, {0}), SubgraphKey(job.model_id, 0, {0})},
        job.expected_latency /*consider job's expected_latency is the model's shortest expected latency*/);
  }

  WorkerId GetModelWorker(ModelId model_id) const override {
    if (w > list_idle_workers_.size())
      return -1;
    else
      return list_idle_workers_[w++];
  }

  WorkerWaitingTime GetWorkerWaitingTime() const override {
    return map;
  }

  void UpdateWorkersWaiting() const override {
    // reset to 0 and recalculate
    for (WorkerId worker_id : list_idle_workers_) {
      map[worker_id] = 0;
    }
    for (auto action: action_){
      map[action.second.GetWorkerId()] += action.first.expected_latency;
    }
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

// <shortest latency, available workers> - model id is assigned in order
struct ModelLevelWithLatencyTestsFixture
    : public testing::TestWithParam<
          std::tuple<std::deque<int64_t>, std::set<int>>> {};

// <shortest latency, target worker for each model, available workers, expected result> - model id is assigned in order
struct HeftFixture // ModelLevelWithLatencyAndFixedWorkerTestsFixture
    : public testing::TestWithParam<
          std::tuple<bool, std::deque<int64_t>, std::deque<int>, std::set<int>, std::deque<int>>> {};

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

TEST_P(ModelLevelWithLatencyTestsFixture, ShortestExepectedLatencyRequestTests) {
  std::deque<int64_t> model_latencies = std::get<0>(GetParam());
  std::set<int> available_workers = std::get<1>(GetParam());
  size_t window_size = 5;

  std::deque<Job> requests;
  for (auto i = 0; i < model_latencies.size(); i++) {
    auto temp = Job(i);
    temp.expected_latency = model_latencies[i]; // consider job's expected_latency is the model's shortest expected latency
    requests.emplace_back(temp);
  }
  
  const int count_requests = requests.size();
  auto sorted_req = std::deque<Job>(requests);
  std::sort(sorted_req.begin(), sorted_req.end(), [](Job a, Job b){
    return a.expected_latency > b.expected_latency;
  });

  MockContext context(available_workers);
  ShortestExpectedLatencyScheduler sel_scheduler(context, std::min(window_size, requests.size()));
  sel_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_models : context.action_) {
    count_scheduled++;
  }

  // min(window_size, # of requested models) should be scheduled
  EXPECT_EQ(count_scheduled,
            std::min(window_size, context.action_.size()));

  // Scheduled requests should be removed from request list.
  EXPECT_EQ(count_requests - count_scheduled, requests.size());

  // scheduled results should me the same as requests descending-sorted by latency (LARGEST shortest subgraph-latency)
  for(int i=0; i<context.action_.size(); i++){
    EXPECT_EQ(context.action_[i].first.model_id, sorted_req[i].model_id);
  }

}

TEST_P(HeftFixture, HEFTRequestTests) {
  bool reserve = std::get<0>(GetParam());
  std::deque<int64_t> model_latencies = std::get<1>(GetParam());
  std::deque<int> target_workers = std::get<2>(GetParam());
  std::set<int> available_workers = std::get<3>(GetParam());
  std::deque<int> expected_scheduling_result = std::get<4>(GetParam());
  size_t window_size = 5;

  assert(target_workers.size() == model_latencies.size());

  std::deque<Job> requests;
  for (auto i = 0; i < model_latencies.size(); i++) {
    auto temp = Job(i);
    temp.job_id = i;
    temp.expected_latency = model_latencies[i]; // consider job's expected_latency is the model's shortest expected latency
    temp.target_worker_id = target_workers[i];
    requests.emplace_back(temp);
  }

  const int count_requests = requests.size();

  MockContext context(available_workers);
  HEFTScheduler heft_scheduler(context, std::min(window_size, requests.size()), reserve);
  heft_scheduler.Schedule(requests);

  int count_scheduled = 0;
  for (auto scheduled_models : context.action_) {
    count_scheduled++;
  }

  // min(window_size, # of requested models) should be scheduled
  EXPECT_EQ(count_scheduled,
            std::min(window_size, context.action_.size()));

  EXPECT_EQ(context.action_.size(), expected_scheduling_result.size());

  // Scheduled requests should be removed from request list.
  EXPECT_EQ(count_requests - count_scheduled, requests.size());

  for(int i=0; i<expected_scheduling_result.size(); i++){
    EXPECT_EQ(context.action_[i].first.model_id, expected_scheduling_result[i]);
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

INSTANTIATE_TEST_SUITE_P(
    ShortestExepectedLatencyRequestTests, ModelLevelWithLatencyTestsFixture,
    testing::Values(std::make_tuple(std::deque<int64_t>{2, 1, 3}, // shortest latency for each model
                                    std::set<int>{0, 1, 2})));

// HEFT: find largest shortest latency job, schedule it if its workers allow it (and update worker waiting time), else don't schdedule it
// case 1: if no workers are available, don't schedule a thing
// case 2: if 3 workers is available, but all jobs want a single worker, schedule only first job with largest subgraph latency, since it becomes busy after scheduling the first job
// case 3: 3 workers available, 3 jobs wanting each of the 3 workers --> schedule by SEL to each worker
// case 4: 2 workers available, 3 jobs wanting each of the 3 workers, but one of them isn't available --> schedule by SEL but don't schedule the idle one
INSTANTIATE_TEST_SUITE_P(
    HEFTRequestTests, HeftFixture,
    testing::Values(
      // case 1 --> 0 scheduled
      std::make_tuple(false, std::deque<int64_t>{2, 1, 3},     // shortest latency for each model
                      std::deque<int>{0, 1, 2},         // fixed worker for each model
                      std::set<int>{},                  // available workers
                      std::deque<int>{}),               // expected scheduling result
      // case 2 --> 2
      std::make_tuple(false, std::deque<int64_t>{2, 1, 3},
                      std::deque<int>{0, 0, 0},
                      std::set<int>{0, 1, 2},
                      std::deque<int>{2}),
      // case 3 --> 2, 0, 1
      std::make_tuple(false, std::deque<int64_t>{2, 1, 3},
                      std::deque<int>{0, 1, 2},
                      std::set<int>{0, 1, 2},
                      std::deque<int>{2, 0, 1}),
      // case 4 --> 2, 0
      std::make_tuple(false, std::deque<int64_t>{2, 1, 3},
                      std::deque<int>{0, 1, 2},
                      std::set<int>{0, 2},
                      std::deque<int>{2, 0}),
      // case 5 --> 1, 2 (non-reserved)
      std::make_tuple(false, std::deque<int64_t>{2, 3, 3},
                      std::deque<int>{0, 0, 2},
                      std::set<int>{0, 1, 2},
                      std::deque<int>{1, 2})
                                ));

}  // namespace test
}  // namespace band

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}