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

load("//band:band.bzl", "band_cc_android_test", "band_cc_library")

exports_files(glob([
    "data/*.tflite",
    "data/*.json",
    "data/*.jpg",
]))

filegroup(
    name = "tfl_models",
    srcs = glob([
        "data/*.tflite",
    ]),
)

filegroup(
    name = "benchmark_jsons",
    srcs = glob([
        "data/*.json",
    ]),
)

filegroup(
    name = "test_images",
    srcs = glob([
        "data/*.jpg",
    ]),
)

band_cc_library(
    name = "test_util",
    srcs = [
        "image_util.cc",
    ],
    hdrs = [
        "image_util.h",
        "test_util.h",
    ],
    copts = [
        "-DSTB_IMAGE_IMPLEMENTATION",
        "-DSTB_IMAGE_WRITE_IMPLEMENTATION",
    ],
    deps = [
        "//band:common",
        "//band/buffer",
        "@com_google_googletest//:gtest",
        "@stblib//:stb_image",
        "@stblib//:stb_image_write",
    ],
)

band_cc_android_test(
    name = "planner_test",
    size = "small",
    srcs = ["planner_test.cc"],
    deps = [
        ":test_util",
        "//band:planner",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["scheduler_test.cc"],
    deps = [
        ":test_util",
        "//band:config",
        "//band:model",
        "//band:scheduler",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "worker_test",
    size = "small",
    srcs = ["worker_test.cc"],
    deps = [
        ":test_util",
        "//band:worker",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "time_test",
    size = "small",
    srcs = ["time_test.cc"],
    deps = [
        ":test_util",
        "//band:time",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "benchmark_test",
    size = "medium",
    srcs = ["tool/benchmark_test.cc"],
    data = [
        "//band/test:tfl_models",
        "//band/test:benchmark_jsons",
    ],
    deps = [
        ":test_util",
        "//band/tool:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "config_builder_test",
    size = "small",
    srcs = ["config_builder_test.cc"],
    deps = [
        ":test_util",
        "//band:config_builder",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "latency_estimator_test",
    size = "small",
    srcs = ["latency_estimator_test.cc"],
    deps = [
        ":test_util",
        "//band:config_builder",
        "//band:latency_estimator",
        "//band:worker",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "cpu_test",
    size = "small",
    srcs = ["cpu_test.cc"],
    data = [],
    deps = [
        "//band/device",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_minimal_test",
    size = "small",
    srcs = ["backend/tfl_minimal_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:tfl_models",
        "//band/test:test_images",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "image_operation_test",
    size = "small",
    srcs = ["image_operation_test.cc"],
    data = [
        "//band/test:data/hippo.jpg",
        "//band/test:data/hippo_crop_256.jpg",
        "//band/test:data/hippo_resize_256.jpg",
    ],
    deps = [
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_subgraph_test",
    size = "enormous",
    srcs = ["backend/tfl_subgraph_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:tfl_models",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c/c_api_test.cc"],
    data = [
        "//band/test:tfl_models",
        "//band/test:test_images",
    ],
    deps = [
        "//band:framework",
        "//band/c:c_api",
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)