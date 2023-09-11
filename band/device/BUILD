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

load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//band:common",
        "@com_google_absl//absl/status",
    ],
)

band_cc_library(
    name = "device",
    srcs = [
        "device.cc",
    ],
    hdrs = [
        "device.h",
    ],
    deps = [
        ":util",
    ],
)

band_cc_library(
    name = "cpu",
    srcs = [
        "cpu.cc",
    ],
    hdrs = [
        "cpu.h",
    ],
    deps = [
        ":util",
    ],
)

band_cc_library(
    name = "thermal",
    srcs = [
        "thermal.cc",
    ],
    hdrs = [
        "thermal.h",
    ],
    deps = [
        ":util",
        "//band:common",
        "//band:config",
    ],
)

band_cc_library(
    name = "frequency",
    srcs = [
        "frequency.cc",
    ],
    hdrs = [
        "frequency.h",
    ],
    deps = [
        ":util",
        "//band:common",
        "//band:config",
    ],
)