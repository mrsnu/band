#ifndef BAND_TRACER_H_
#define BAND_TRACER_H_

#include <mutex>
#include <unordered_map>

#include "band/common.h"
#include "chrome_tracer/tracer.h"

namespace band {
class JobTracer : public chrome_tracer::ChromeTracer {
 public:
  static JobTracer& Get();
  void BeginSubgraph(const Job& job);
  void EndSubgraph(const Job& job);
  void AddWorker(DeviceFlag device_flag, size_t id);
  void Dump(std::string path) const;

 private:
  JobTracer();
  JobTracer(const JobTracer&) = delete;

  std::string GetStreamName(size_t id) const;
  std::string GetJobName(const Job& job) const;

  std::mutex mtx_;
  std::map<size_t, std::string> id_to_streams_;
  std::unordered_map<std::pair<size_t, BitMask>, int32_t, JobIdBitMaskHash>
      job_to_handle_;
};
}  // namespace band

#ifdef BAND_TRACE
#define BAND_TRACER_ADD_WORKER(device_flag, id)        \
  do {                                                 \
    band::JobTracer::Get().AddWorker(device_flag, id); \
  } while (0)

#define BAND_TRACER_BEGIN_SUBGRAPH(job)        \
  do {                                         \
    band::JobTracer::Get().BeginSubgraph(job); \
  } while (0)

#define BAND_TRACER_END_SUBGRAPH(job)        \
  do {                                       \
    band::JobTracer::Get().EndSubgraph(job); \
  } while (0)

#define BAND_TRACER_DUMP(path)         \
  do {                                 \
    band::JobTracer::Get().Dump(path); \
  } while (0)

#else
#define BAND_TRACER_ADD_STREAM(...)
#define BAND_TRACER_ADD_WORKER(...)
#define BAND_TRACER_BEGIN_SUBGRAPH(...)
#define BAND_TRACER_END_SUBGRAPH(...)
#define BAND_TRACER_DUMP(...)
#endif

#endif  // BAND_TRACER_H_