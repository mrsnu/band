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

#ifndef BAND_TIME_H_
#define BAND_TIME_H_

#include <cstdint>

namespace band {
namespace time {
uint64_t NowMicros();
uint64_t NowNanos();
void SleepForMicros(uint64_t micros);
}  // namespace time
}  // namespace band
#endif  // BAND_TIME_H_
