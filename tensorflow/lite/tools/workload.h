#ifndef TENSORFLOW_LITE_TOOLS_WORKLOAD_H_
#define TENSORFLOW_LITE_TOOLS_WORKLOAD_H_

#include <map>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/util.h"

namespace tflite {

struct Frame {
  struct ModelRequest {
    ModelRequest(Job job, int id, int count, std::vector<int> dependency);
    const Job job;
    const int id;
    const int count;
    const std::vector<int> dependency;
  };

  std::map<int, ModelRequest> requests;
};

class Workload {
 public:
  Workload();
  Workload(std::vector<Frame> frames);

  TfLiteStatus ExecuteFrame(tflite::Interpreter* interpreter);
  void Reset();
  bool IsFinished() const;
  size_t GetCurrentFrame() const { return current_frame_; }

 private:
  std::vector<Job> GetNextRequests(const Frame& frame,
                                   std::set<int>& resolved_requests) const;

  size_t current_frame_ = 0;
  std::vector<Frame> frames_;
};

TfLiteStatus ParseWorkloadFromJson(std::string json_fname,
                                   std::map<int, ModelConfig>& model_config,
                                   Workload& workload);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_WORKLOAD_H_