#ifndef BAND_TRACER_H_
#define BAND_TRACER_H_

#include <chrome_tracer/tracer.h>

#include "band/common.h"

namespace band {
namespace /* unnamed */ {
class JobTracer {
 public:
  static JobTracer& Get();
  void AddJob(const Job& job);

 private:
  JobTracer();
  JobTracer(const JobTracer&) = delete;

  chrome_tracer::ChromeTracer tracer_;
};
}  // namespace
}  // namespace band

#ifdef BAND_TRACE
BAND_TRACER_ADD_STREAM(name)

#elif
#define BAND_TRACER_ADD_STREAM(name) ...

#endif