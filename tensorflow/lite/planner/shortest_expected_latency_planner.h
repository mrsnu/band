#ifndef TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
#define TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/planner/planner.h"

namespace tflite {
namespace impl {

class ShortestExpectedLatencyPlanner : public Planner {
 public:
  explicit ShortestExpectedLatencyPlanner(Interpreter* interpreter)
      : Planner(interpreter) {
    planner_thread_ = std::thread([this] { this->Plan(); });
  }
  void Plan() override;
  bool NeedProfile() override;

 private:
  // Return a pair of the subgraph idx that leads to the shortest final
  // latency, and that final latency value.
  // Note that the returned subgraph may only cover a subset of the remaining
  // ops, but the latency value is calculated with all subgraphs leading to
  // the final op (of the model) in mind.
  std::pair<int, int64_t> GetShortestLatency(
      int model_id, std::set<int> resolved_output, int64_t start_time,
      std::map<TfLiteDeviceFlags, int64_t>& device_waiting,
      TfLiteDeviceFlags preceded_device = kTfLiteNumDevices);

  /* private methods related to subgraph scheduling */
  // divide the given subgraphs into groups that share the same start/end idxs
  // e.g., {(0,10): [1,3], (0,20): [2,4]}
  std::map<std::pair<std::set<int>, std::set<int>>, std::vector<int>>
  GroupByStartEndIdx(std::vector<int> subgraph_indices);

  // return subgraph indices for model_id and start_idx,
  // excluding subgraphs on preceded_device
  std::vector<int> GetSubgraphCandidates(int model_id, std::set<int> resolved_output,
                                         TfLiteDeviceFlags preceded_device);

  // return the shortest subgraph out of given subgraphs, when the start time
  // and per-device waiting times are taken into account
  std::pair<int, int64_t> GetShortestSubgraphIndex(
      std::vector<int> subgraph_indices, int64_t start_time,
      std::map<TfLiteDeviceFlags, int64_t>& device_waiting);
};

}  // namespace impl
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PLANNER_SHORTEST_EXPECTED_LATENCY_PLANNER_H_
