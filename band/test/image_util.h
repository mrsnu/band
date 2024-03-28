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

#ifndef BAND_TEST_IMAGE_UTIL_H_
#define BAND_TEST_IMAGE_UTIL_H_

#include <memory>
#include <string>

namespace band {
class Buffer;
namespace test {

std::shared_ptr<Buffer> LoadImage(const std::string& filename);
std::tuple<unsigned char*, int, int> LoadRGBImageRaw(
    const std::string& filename);
void SaveImage(const std::string& filename, const Buffer& buffer);

}  // namespace test
}  // namespace band

#endif