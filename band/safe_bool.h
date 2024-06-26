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

#ifndef BAND_SAFE_BOOL_H_
#define BAND_SAFE_BOOL_H_

#include <condition_variable>
#include <mutex>
namespace band {
class SafeBool {
 public:
  SafeBool() = default;
  ~SafeBool() = default;

  void notify();
  bool wait();
  void terminate();

 private:
  mutable std::mutex m;
  bool flag = false;
  bool exit = false;
  std::condition_variable c;
};

}  // namespace band

#endif  // BAND_SAFE_BOOL_H_
