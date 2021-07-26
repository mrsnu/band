load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = [
    "-Wall",
] + tflite_copts()

cc_library(
    name = "cpu",
    srcs = [
        "cpu.cc",
    ],
    hdrs = [
        "cpu.h",
    ],
    deps = [
        ":util",
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "gpu",
    srcs = [
        "gpu.cc",
    ],
    hdrs = [
        "gpu.h",
    ],
    deps = [
        ":util",
        "//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)