#include "job_tracer.h"

namespace band {
JobTracer::JobTracer() : chrome_tracer::ChromeTracer("Band") {}

std::string JobTracer::GetStreamName(size_t id) const {
  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    std::cerr << "The given stream id does not exists." << std::endl;
    abort();
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
  BeginEvent(GetStreamName(job.subgraph_key.GetWorkerId()), GetJobName(job));
}

void JobTracer::EndSubgraph(const Job& job) {
  EndEvent(GetStreamName(job.subgraph_key.GetWorkerId()), GetJobName(job),
           job.ToJson());
}

void JobTracer::AddWorker(BandDeviceFlags device_flag, size_t id) {
  std::string stream_name = std::string("(") + BandDeviceGetName(device_flag) +
                            "Worker ," + std::to_string(id) + ")";

  if (id_to_streams_.find(id) != id_to_streams_.end()) {
    std::cerr << "The given stream id already exists." << std::endl;
    abort();
  }
  id_to_streams_[id] = stream_name;
  chrome_tracer::ChromeTracer::AddStream(stream_name);
}

void JobTracer::Dump(std::string path) const {
  return chrome_tracer::ChromeTracer::Dump(path);
}
}  // namespace band