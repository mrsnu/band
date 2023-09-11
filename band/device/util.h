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

#ifndef BAND_DEVICE_UTIL_H_
#define BAND_DEVICE_UTIL_H_

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "band/common.h"

#if defined(__ANDROID__) || defined(__IOS__)
#define BAND_IS_MOBILE 1
#else
#define BAND_IS_MOBILE 0
#endif

namespace band {
namespace device {

template <typename T>
absl::StatusOr<T> TryRead(std::vector<std::string> paths,
                          std::vector<float> multipliers = {});

absl::StatusOr<size_t> TryReadSizeT(std::vector<std::string> paths,
                                    std::vector<float> multipliers = {1.f});
absl::StatusOr<double> TryReadDouble(std::vector<std::string> paths,
                                     std::vector<float> multipliers = {1.f});

absl::StatusOr<std::vector<size_t>> TryReadSizeTs(
    std::vector<std::string> paths, std::vector<float> multipliers = {1.f});
absl::StatusOr<std::vector<double>> TryReadDoubles(
    std::vector<std::string> paths, std::vector<float> multipliers = {1.f});

std::vector<std::string> ListFilesInPath(const char* path);
std::vector<std::string> ListFilesInPathPrefix(const char* path,
                                               const char* prefix);
std::vector<std::string> ListFilesInPathSuffix(const char* path,
                                               const char* suffix);
std::vector<std::string> ListDirectoriesInPath(const char* path);
std::vector<std::string> ListDirectoriesInPathPrefix(const char* path,
                                                     const char* prefix);
std::vector<std::string> ListDirectoriesInPathSuffix(const char* path,
                                                     const char* suffix);

bool IsFileAvailable(std::string path);
void Root();
std::string RunCommand(const std::string& command);
absl::StatusOr<std::string> GetDeviceProperty(const std::string& property);

}  // namespace device
}  // namespace band

#endif  // BAND_DEVICE_UTIL_H_
