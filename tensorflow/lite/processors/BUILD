load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = [
    "-Wall",
] + tflite_copts()

cc_library(
    name = "processor",
    srcs = [
        "cpu.cc",
        "generic.cc",
        "gpu.cc",
        "util.cc",
        "processor.cc",
    ],
    hdrs = [
        "cpu.h",
        "generic.h",
        "gpu.h",
        "util.h",
        "processor.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
    ],
)