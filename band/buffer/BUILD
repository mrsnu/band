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

load(
    "//band:band.bzl",
    "band_cc_library",
)

package(
    default_visibility = ["//band:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "buffer",
    srcs = [
        "buffer.cc",
        "buffer_processor.cc",
        "common_operator.cc",
        "image_operator.cc",
        "image_processor.cc",
        "libyuv_image_operator.cc",
        "operator.cc",
    ],
    hdrs = [
        "buffer.h",
        "buffer_processor.h",
        "common_operator.h",
        "image_operator.h",
        "image_processor.h",
        "libyuv_image_operator.h",
        "operator.h",
    ],
    deps = [
        "//band:common",
        "//band:interface",
        "@libyuv",
    ],
)
