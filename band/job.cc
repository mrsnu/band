#include "band/job.h"

namespace band {

std::size_t JobIdBitMaskHash::operator()(
    const std::pair<int, BitMask>& p) const {
  auto hash_func = std::hash<int>();
  return hash_func(p.first) ^ hash_func(p.second.to_ullong());
}

bool Job::IsInitialized() const {
  return (model_id != -1) && (job_id != -1) && (input_handle != -1) &&
         (output_handle != -1);
}

std::string Job::ToJson() const {
  return "{\"enqueue_time\":" + std::to_string(enqueue_time) +
         ",\"invoke_time\":" + std::to_string(invoke_time) +
         ",\"end_time\":" + std::to_string(end_time) +
         ",\"profiled_execution_time\":" +
         std::to_string(profiled_execution_time) +
         ",\"expected_execution_time\":" +
         std::to_string(expected_execution_time) +
         ",\"expected_latency\":" + std::to_string(expected_latency) +
         ",\"slo_us\":" + std::to_string(slo_us) +
         ",\"model_id\":" + std::to_string(model_id) +
         ",\"unit_indices\":" + subgraph_key.GetUnitIndicesString() +
         ",\"job_id\":" + std::to_string(job_id) + "}";
}

}  // namespace band