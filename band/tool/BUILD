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
    "clean_dep",
    "band_cc_library",
)
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "band_benchmark",
    srcs = [
        "band_benchmark.cc",
    ],
    linkopts = select({
        clean_dep("//band:android"): [
            "-pie",
            "-lm",
            "-Wl,--rpath=/data/local/tmp/",
        ],
        "//conditions:default": [],
    }),
    linkstatic = True,
    deps = [
        ":benchmark",
    ],
)


band_cc_library(
    name = "benchmark",
    srcs = [
        "benchmark.cc",
        "benchmark_profiler.cc"
    ],
    hdrs = [
        "benchmark.h",
        "benchmark_config.h",
        "benchmark_profiler.h"
    ],
    deps = [
        "//band:framework",
        "//band:config_builder",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)