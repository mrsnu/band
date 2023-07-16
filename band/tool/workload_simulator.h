#ifndef BAND_TOOLS_WORKLOAD_GENERATOR_H_
#define BAND_TOOLS_WORKLOAD_GENERATOR_H_

#include <map>
#include <vector>

#include "band/common.h"
#include "band/config.h"
#include "band/model.h"

namespace band {

struct Frame {
  struct ModelRequest {
    ModelRequest(Job job, int id, int count, std::vector<int> parent_requests);
    const Job job;
    const int id;
    const int count;
    const std::vector<int> parent_requests;
  };

  // job id to requests
  std::map<int, ModelRequest> requests;
};

class WorkloadSimulator {
 public:
  WorkloadSimulator();
  WorkloadSimulator(std::vector<Frame> frames);

  absl::Status ExecuteCurrentFrame(
      Engine* engine, const std::vector<Tensors>& model_input_tensors = {},
      const std::vector<Tensors>& model_output_tensors = {});
  void Reset();
  bool IsFinished() const;
  size_t GetNumFrames() const { return frames_.size(); }
  size_t GetCurrentFrame() const { return current_frame_; }

 private:
  std::vector<Job> GetNextRequests(const Frame& frame,
                                   std::set<int>& resolved_requests) const;

  size_t current_frame_ = 0;
  std::vector<Frame> frames_;
};

absl::Status ParseWorkloadFromJson(std::string json_fname,
                                   std::map<int, ModelConfig>& model_config,
                                   WorkloadSimulator& workload);

}  // namespace band

#endif  // BAND_TOOLS_WORKLOAD_SIMULATOR_H_