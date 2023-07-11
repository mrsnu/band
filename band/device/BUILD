load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "device",
    srcs = [
        "cpu.cc",
        "generic.cc",
        "gpu.cc",
        "processor.cc",
        "util.cc",
    ],
    hdrs = [
        "cpu.h",
        "generic.h",
        "gpu.h",
        "processor.h",
        "util.h",
    ],
    deps = [
        "//band:common",
        "@com_google_absl//absl/status",
    ],
)
