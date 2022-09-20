#include "tensorflow/lite/planner/thermal_aware_scheduler.h"

#include "tensorflow/lite/profiling/time.h"

namespace tflite {
namespace impl {

void ThermalAwareScheduler::Schedule(JobQueue& requests) {
  // TODO : implement
}

int64_t ThermalAwareScheduler::GetCurrentTemperature() {
  // TODO : implement
  return INT_MAX;
}

void ThermalAwareScheduler::UpdateExpectedLatency(JobQueue& requests,
                                                     int window_size) {
  // TODO : implement
}

void ThermalAwareScheduler::UpdateExpectedHeatGeneration(JobQueue& requests,
                                                     int window_size) {
  // TODO : implement
}

}  // namespace impl
}  // namespace tflite
