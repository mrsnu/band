#include "band/context.h"

namespace Band {
Context::Context(ErrorReporter* error_reporeter)
    : error_reporter_(error_reporeter) {}

void Context::UpdateWorkerWaitingTime() const {}

const WorkerWaitingTime& Context::GetWorkerWaitingTime() const {}

std::set<WorkerId> Context::GetIdleWorkers() const {}

SubgraphKey Context::GetModelSubgraphKey(ModelId model_id,
                                         WorkerId worker_id) const {}

bool Context::IsEnd(const SubgraphKey& key) const {}

BandStatus Context::Invoke(const SubgraphKey& key) {}

const ModelSpec* Context::GetModelSpec(ModelId model_id) {}

const ModelConfig* Context::GetModelConfig(ModelId model_id) const {}
WorkerId Context::GetModelWorker(ModelId model_id) const {}

std::pair<SubgraphKey, int64_t> Context::GetShortestLatency(
    int model_id, std::set<int> resolved_tensors, int64_t start_time,
    const std::map<WorkerId, int64_t>& worker_waiting,
    SubgraphKey preceded_subgraph_index) const {}

std::pair<std::vector<SubgraphKey>, int64_t>
Context::GetShortestLatencyWithUnitSubgraph(
    int model_id, int start_unit_idx,
    const std::map<WorkerId, int64_t>& worker_waiting) const {}

std::pair<std::vector<SubgraphKey>, int64_t>
Context::GetSubgraphWithShortestLatency(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting) const {}

SubgraphKey Context::GetSubgraphIdxSatisfyingSLO(
    Job& job, const std::map<WorkerId, int64_t>& worker_waiting,
    const std::set<WorkerId>& idle_workers) const {}

void Context::UpdateLatency(const SubgraphKey& key, int64_t latency) {}

int64_t Context::GetProfiled(const SubgraphKey& key) const {}

int64_t Context::GetExpected(const SubgraphKey& key) const {}

void Context::Trigger() {}

JobId Context::EnqueueRequest(Job job, bool push_front) {}

std::vector<JobId> Context::EnqueueBatch(std::vector<Job> jobs,
                                         bool push_front) {}

void Context::PrepareReenqueue(Job& job) {}

void Context::EnqueueFinishedJob(Job& job) {}

Worker* Context::GetWorker(WorkerId id) {}

BandStatus Context::TryCopyInputTensors(const Job& job) { return kBandOk; }

BandStatus Context::TryCopyOutputTensors(const Job& job) { return kBandOk; }
}  // namespace Band