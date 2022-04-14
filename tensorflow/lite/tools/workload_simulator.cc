#include "tensorflow/lite/tools/workload_simulator.h"

#include <json/json.h>

#include <algorithm>
#include <set>

#include "tensorflow/lite/config.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
Frame::ModelRequest::ModelRequest(Job job, int id, int count,
                                  std::vector<int> parent_requests)
    : job(job), id(id), count(count), parent_requests(parent_requests) {}

WorkloadSimulator::WorkloadSimulator() {}

WorkloadSimulator::WorkloadSimulator(std::vector<Frame> frames)
    : frames_(frames) {}

TfLiteStatus WorkloadSimulator::ExecuteCurrentFrame(
    tflite::Interpreter* interpreter, 
    const std::vector<Tensors>& model_input_tensors,
    const std::vector<Tensors>& model_output_tensors) {
  if (IsFinished()) {
    return kTfLiteError;
  }

  auto& current_frame = frames_[current_frame_++];
  std::set<int> resolved_requests;

  while (true) {
    std::vector<Job> next_batch =
        GetNextRequests(current_frame, resolved_requests);
    if (next_batch.size() == 0) break;

    std::vector<Tensors> inputs, outputs;

    if (model_input_tensors.size() && model_output_tensors.size()) {
      for(auto& job : next_batch) {
        inputs.push_back(model_input_tensors[job.model_id]);
        outputs.push_back(model_output_tensors[job.model_id]);
      }
    }

    interpreter->InvokeModelsSync(next_batch, inputs, outputs);
  };

  return kTfLiteOk;
}

void WorkloadSimulator::Reset() { current_frame_ = 0; }

bool WorkloadSimulator::IsFinished() const {
  return current_frame_ >= frames_.size();
}

std::vector<Job> WorkloadSimulator::GetNextRequests(
    const Frame& frame, std::set<int>& resolved_requests) const {
  std::set<int> current_requests;
  bool requires_update = true;
  while (requires_update) {
    requires_update = false;

    for (auto& request_it : frame.requests) {
      // skip already executed ones
      if (resolved_requests.find(request_it.first) != resolved_requests.end()) {
        continue;
      }

      auto& request = request_it.second;
      if (std::includes(resolved_requests.begin(), resolved_requests.end(),
                        request.parent_requests.begin(),
                        request.parent_requests.end())) {
        // re-iterate if there is a zero-sized request
        if (request.count == 0) {
          requires_update = true;
          resolved_requests.insert(request_it.first);
        } else {
          current_requests.insert(request_it.first);
        }
      }
    }
  };

  std::vector<Job> next_requests;
  for (int request_id : current_requests) {
    auto& request = frame.requests.at(request_id);
    for (int i = 0; i < request.count; i++) {
      next_requests.push_back(request.job);
    }

    resolved_requests.insert(request_id);
  }

  return next_requests;
}

TfLiteStatus ParseWorkloadFromJson(std::string json_fname,
                                   std::map<int, ModelConfig>& model_config,
                                   WorkloadSimulator& workload) {
  std::ifstream config(json_fname, std::ifstream::binary);

  if (!config.is_open()) {
    TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Check if %s exists in workload.", json_fname.c_str());
    return kTfLiteError;
  }

  std::map<std::string, int> model_fname_to_id;

  auto get_basename = [](std::string path) {
    return path.substr(path.find_last_of("/\\") + 1);
  };

  for (auto& it : model_config) {
    model_fname_to_id.insert({get_basename(it.second.model_fname), it.first});
  }

  Json::Value root;
  config >> root;

  std::vector<Frame> frames(root.size());
  for (int i = 0; i < root.size(); i++) {
    auto frame = root[i];

    for (Json::Value::const_iterator itr = frame.begin(); itr != frame.end();
         itr++) {
      // workaround (str to int) as js does't support non-string key
      int request_id = stoi(itr.key().asString());
      auto& request = *itr;

      TF_LITE_ENSURE_STATUS(ValidateJsonConfig(
          request, {"model", "count", "dependency"}));

      std::string model_name = request["model"].asString();
      if (model_fname_to_id.find(model_name) == model_fname_to_id.end()) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Check if %s exists in model list.", model_name.c_str());
        return kTfLiteError;
      }

      std::vector<int> dependency;
      for (int j = 0; j < request["dependency"].size(); j++) {
        dependency.push_back(request["dependency"][j].asInt());
      }

      frames[i].requests.insert(
          {request_id,
           Frame::ModelRequest(Job(model_fname_to_id[model_name]), request_id,
                               request["count"].asInt(), dependency)});
    }
  }

  workload = WorkloadSimulator(frames);

  return kTfLiteOk;
}
}  // namespace tflite
