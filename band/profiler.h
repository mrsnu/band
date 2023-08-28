/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_PROFILER_H_
#define BAND_PROFILER_H_

#include <chrono>
#include <vector>

namespace band {

class Profiler {
 public:
  size_t BeginEvent();
  void EndEvent(size_t event_handle);
  size_t GetNumEvents() const;

  template <typename T>
  double GetElapsedTimeAt(size_t index) const {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");
    if (timeline_vector_.size() > index) {
      return std::max<double>(
          std::chrono::duration_cast<T>(timeline_vector_[index].second -
                                        timeline_vector_[index].first)
              .count(),
          0);
    } else
      return 0;
  }

  template <typename T>
  double GetAverageElapsedTime() const {
    static_assert(is_chrono_duration<T>::value,
                  "T must be a std::chrono::duration");

    double accumulated_time = 0;
    for (size_t i = 0; i < timeline_vector_.size(); i++) {
      accumulated_time += GetElapsedTimeAt<T>(i);
    }

    if (timeline_vector_.size() == 0) {
      return 0;
    }

    return accumulated_time / timeline_vector_.size();
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
      timeline_vector_;
};
}  // namespace band
#endif