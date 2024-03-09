#include "band/job_tracer.h"

#include "band/logger.h"
#include "job_tracer.h"

namespace band {
JobTracer::JobTracer()
#ifdef __ANDROID__
    : chrome_tracer::ChromeStreamTracer("/data/local/tmp/splash/log.json")
#else
    : chrome_tracer::ChromeStreamTracer("log.json")
#endif
{
}

std::string JobTracer::GetStreamName(size_t id) const {
  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    band::Logger::Log(BAND_LOG_INFO, "The given worker id does not exists. %zd",
                      id);
    return "";
  }
  return id_to_streams_.at(id);
}

std::string JobTracer::GetJobName(const Job& job) const {
  std::string model_name;
  if (job.model_id == -1) {
    model_name = "(Idle";
  } else if (job.model_fname.empty()) {
    model_name = "(Model " + std::to_string(job.model_id);
  } else {
    model_name = job.model_fname;
  }
  return model_name + ", JobId " + std::to_string(job.job_id) + ")";
}

std::string JobTracer::GetThreadStreamName() {
  std::thread::id thread_id = std::this_thread::get_id();
  if (thread_to_stream_.find(thread_id) == thread_to_stream_.end()) {
    size_t hashed_id = std::hash<std::thread::id>{}(thread_id);
    std::string id =
        std::string("(Thread ") + std::to_string(hashed_id) + std::string(")");
    thread_to_stream_[thread_id] = id;
    chrome_tracer::ChromeStreamTracer::AddStream(id);
  }
  return thread_to_stream_[thread_id];
}

JobTracer& JobTracer::Get() {
  static JobTracer tracer;
  return tracer;
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
    band::Logger::Log(BAND_LOG_INFO,
                      "The given job does not exists. (id:%d, unit_indices:%s)",
                      job.job_id,
                      job.subgraph_key.GetUnitIndicesString().c_str());
  }
}

void JobTracer::AddWorker(DeviceFlag device_flag, size_t id) {
  std::string stream_name = std::string("(") + ToString(device_flag) +
                            "Worker ," + std::to_string(id) + ")";

  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    id_to_streams_[id] = stream_name;
    chrome_tracer::ChromeStreamTracer::AddStream(stream_name);
  } else {
    band::Logger::Log(BAND_LOG_INFO, "The given worker id already exists. %zd",
                      id);
  }
}

int32_t JobTracer::BeginThreadEvent(std::string event) {
  return chrome_tracer::ChromeStreamTracer::BeginEvent(GetThreadStreamName(),
                                                       event);
}
void JobTracer::EndThreadEvent(int32_t handle, std::string args) {
  return chrome_tracer::ChromeStreamTracer::EndEvent(GetThreadStreamName(),
                                                     handle, args);
}
}  // namespace band