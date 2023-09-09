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

#include "job_tracer.h"

#include "band/logger.h"

namespace band {
JobTracer::JobTracer() : chrome_tracer::ChromeTracer("Band") {}

std::string JobTracer::GetStreamName(size_t id) const {
  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    BAND_LOG(LogSeverity::kWarning, "The given worker id does not exists. %zd",
             id);
    return "";
  }
  return id_to_streams_.at(id);
}

std::string JobTracer::GetJobName(const Job& job) const {
  std::string model_name =
      "(Model " +
      (job.model_fname == "" ? std::to_string(job.model_id) : job.model_fname);
  return model_name + ", JobId " + std::to_string(job.job_id) + ")";
}

JobTracer& JobTracer::Get() {
  static JobTracer* tracer = new JobTracer;
  return *tracer;
}

void JobTracer::BeginSubgraph(const Job& job) {
  std::unique_lock<std::mutex> lock(mtx_);
  job_to_handle_[{job.job_id, job.subgraph_key.GetUnitIndices()}] = BeginEvent(
      GetStreamName(job.subgraph_key.GetWorkerId()), GetJobName(job));
}

void JobTracer::EndSubgraph(const Job& job) {
  std::unique_lock<std::mutex> lock(mtx_);
  std::pair<size_t, BitMask> key = {job.job_id,
                                    job.subgraph_key.GetUnitIndices()};
  if (job_to_handle_.find(key) != job_to_handle_.end()) {
    EndEvent(GetStreamName(job.subgraph_key.GetWorkerId()),
             job_to_handle_.at(key), job.ToJson());
  } else {
    BAND_LOG(LogSeverity::kInfo,
             "The given job does not exists. (id:%d, unit_indices:%s)",
             job.job_id, job.subgraph_key.GetUnitIndicesString().c_str());
  }
}

void JobTracer::AddWorker(DeviceFlag device_flag, size_t id) {
  std::string stream_name = std::string("(") + ToString(device_flag) +
                            "Worker ," + std::to_string(id) + ")";

  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    id_to_streams_[id] = stream_name;
    chrome_tracer::ChromeTracer::AddStream(stream_name);
  } else {
    BAND_LOG(LogSeverity::kInfo, "The given worker id already exists. %zd", id);
  }
}

void JobTracer::Dump(std::string path) const {
  return chrome_tracer::ChromeTracer::Dump(path);
}
}  // namespace band