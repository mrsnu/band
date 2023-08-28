# Copyright 2023 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
#   Single-file C++ image decoding and encoding libraries

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT license

exports_files(["LICENSE"])

cc_library(
    name = "stb_image",
    hdrs = ["stb_image.h"],
    copts = [
        "-Wno-unused-function",
        "$(STACK_FRAME_UNLIMITED)",
    ],
    includes = ["."],
)

cc_library(
    name = "stb_image_write",
    hdrs = ["stb_image_write.h"],
    includes = ["."],
)
