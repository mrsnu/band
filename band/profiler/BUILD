load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "profiler",
    hdrs = [
        "profiler.h",
    ],
    deps = [
        "//band:common",
    ],
)

band_cc_library(
    name = "latency_profiler",
    srcs = [
        "latency_profiler.cc",
    ],
    hdrs = [
        "latency_profiler.h",
    ],
    deps = [
        ":profiler",
    ],
)

band_cc_library(
    name = "thermal_profiler",
    srcs = [
        "thermal_profiler.cc",
    ],
    hdrs = [
        "thermal_profiler.h",
    ],
    deps = [
        ":profiler",
        "//band/device:thermal",
    ],
)

band_cc_library(
    name = "profilers",
    deps = [
        ":latency_profiler",
        ":thermal_profiler",
    ]
)