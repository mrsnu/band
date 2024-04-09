#include "job_tracer.h"

#include "band/logger.h"

/**
 * @brief The `JobTracer` class provides functionality for tracing job events and generating Chrome tracing files.
 */
namespace band {
/**
 * @brief Constructs a `JobTracer` object with the given name.
 */
JobTracer::JobTracer() : chrome_tracer::ChromeTracer("Band") {}

/**
 * @brief Retrieves the stream name associated with the given worker ID.
 * 
 * @param id The worker ID.
 * @return The stream name associated with the worker ID, or an empty string if the worker ID does not exist.
 */
std::string JobTracer::GetStreamName(size_t id) const {
  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    BAND_LOG(LogSeverity::kWarning, "The given worker id does not exists. %zd",
             id);
    return "";
  }
  return id_to_streams_.at(id);
}

/**
 * @brief Retrieves the job name associated with the given job.
 * 
 * @param job The job object.
 * @return The job name.
 */
std::string JobTracer::GetJobName(const Job& job) const {
  std::string model_name =
      "(Model " +
      (job.model_fname == "" ? std::to_string(job.model_id) : job.model_fname);
  return model_name + ", JobId " + std::to_string(job.job_id) + ")";
}

/**
 * @brief Retrieves the singleton instance of the `JobTracer` class.
 * 
 * @return The singleton instance of the `JobTracer` class.
 */
JobTracer& JobTracer::Get() {
  static JobTracer* tracer = new JobTracer;
  return *tracer;
}

/**
 * @brief Begins a subgraph event for the given job.
 * 
 * @param job The job object.
 */
void JobTracer::BeginSubgraph(const Job& job) {
  std::unique_lock<std::mutex> lock(mtx_);
  job_to_handle_[{job.job_id, job.subgraph_key.GetUnitIndices()}] = BeginEvent(
      GetStreamName(job.subgraph_key.GetWorkerId()), GetJobName(job));
}

/**
 * @brief Ends a subgraph event for the given job.
 * 
 * @param job The job object.
 */
void JobTracer::EndSubgraph(const Job& job) {
  std::unique_lock<std::mutex> lock(mtx_);
  std::pair<size_t, BitMask> key = {job.job_id,
                                    job.subgraph_key.GetUnitIndices()};
  if (job_to_handle_.find(key) != job_to_handle_.end()) {
    EndEvent(GetStreamName(job.subgraph_key.GetWorkerId()),
             job_to_handle_.at(key), job.ToJson());
  } else {
    BAND_LOG(LogSeverity::kWarning,
             "The given job does not exists. (id:%d, unit_indices:%s)",
             job.job_id, job.subgraph_key.GetUnitIndicesString().c_str());
  }
}

/**
 * @brief Adds a worker with the given device flag and ID.
 * 
 * @param device_flag The device flag of the worker.
 * @param id The ID of the worker.
 */
void JobTracer::AddWorker(DeviceFlag device_flag, size_t id) {
  std::string stream_name = std::string("(") + ToString(device_flag) +
                            "Worker ," + std::to_string(id) + ")";

  if (id_to_streams_.find(id) == id_to_streams_.end()) {
    id_to_streams_[id] = stream_name;
    chrome_tracer::ChromeTracer::AddStream(stream_name);
  } else {
    BAND_LOG(LogSeverity::kWarning, "The given worker id already exists. %zd", id);
  }
}

/**
 * @brief Dumps the Chrome tracing data to the specified path.
 * 
 * @param path The path to dump the Chrome tracing data.
 */
void JobTracer::Dump(std::string path) const {
  return chrome_tracer::ChromeTracer::Dump(path);
}
}  // namespace band