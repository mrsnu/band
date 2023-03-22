#include "job_tracer.h"

JobTracer& band::JobTracer::Get() {
  static JobTracer* tracer = new tracer;
  return *tracer;
}

void band::JobTracer::AddJob(const Job& job) {
  // TODO: insert return statement here
}