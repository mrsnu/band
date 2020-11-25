#ifndef TENSORFLOW_LITE_PROFILING_TIME_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_TIME_PROFILER_H_

#include <vector>
#include <cmath>
#include <chrono>
#include <type_traits>

#include "tensorflow/lite/core/api/profiler.h"

namespace tflite {
namespace profiling {


// Profiler class for average latency computation
class TimeProfiler : public tflite::Profiler {
 public:
  TimeProfiler();
  ~TimeProfiler() override = default;

  uint32_t BeginEvent(const char* tag, EventType event_type,
                      int64_t event_metadata1,
                      int64_t event_metadata2) override;

  void EndEvent(uint32_t event_handle) override;

  void ClearRecords();
  size_t GetNumInvokeTimelines() const;

  template <typename T>
  uint64_t GetElapsedTimeAt(size_t index) {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");
    if (invoke_timeline_vector_.size() > index) {
      return std::chrono::duration_cast<T>(
        invoke_timeline_vector_[index].second - 
        invoke_timeline_vector_[index].first).count();
    } else
      return 0;
  }

  template <typename T>
  uint64_t GetAverageElapsedTime() {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");

    uint64_t accumulated_time = 0;
    for (size_t i = 0; i < invoke_timeline_vector_.size(); i++) {
      accumulated_time += GetElapsedTimeAt<T>(i);
    }

    return accumulated_time / invoke_timeline_vector_.size();
  }

  template <typename T>
  double GetStandardDeviation() {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");

    uint64_t average = GetAverageElapsedTime<T>();
    double standard_deviation = 0;
    for (size_t i = 0; i < invoke_timeline_vector_.size(); i++) {
      standard_deviation += pow(GetElapsedTimeAt<T>(i) - average, 2);
    }

    return std::sqrt(standard_deviation / invoke_timeline_vector_.size());
  }

  private:
  template <typename T>
  struct is_chrono_duration {
    static constexpr bool value = false;
};

  template <typename Rep, typename Period>
  struct is_chrono_duration<std::chrono::duration<Rep, Period>> {
      static constexpr bool value = true;
  };

  std::vector<std::pair<std::chrono::system_clock::time_point,
                        std::chrono::system_clock::time_point>>
      invoke_timeline_vector_;
};

}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_TIME_PROFILER_H_