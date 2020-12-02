load("//tensorflow/lite:build_def.bzl", "tflite_copts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "jute_json_parser",
    hdrs = [
        "jute.h",
    ],
    copts = common_copts,
    deps = [
        "//tensorflow/lite/c:common",
    ],
    srcs = [
        "jute.cpp",
    ],
)
