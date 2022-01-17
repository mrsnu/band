load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = [
    "-Wall",
] + tflite_copts()

cc_library(
    name = "tensor_memory_pool",
    srcs = ["tensor_memory_pool.cc"],
    hdrs = [
      "tensor_memory_pool.h",
      "simple_memory_pool.h"
      ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)

cc_test(
    name = "tensor_memory_pool_test",
    size = "small",
    srcs = ["tensor_memory_pool_test.cc"],
    tags = [
        "tflite_not_portable_android",
    ],
    deps = [
        ":tensor_memory_pool",
        "//tensorflow/lite/testing:util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "tensor_ring_buffer",
    srcs = ["tensor_ring_buffer.cc"],
    hdrs = ["tensor_ring_buffer.h"],
    copts = common_copts,
    deps = [
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
    ],
)