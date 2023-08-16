load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "profiler",
    srcs = [
        "profiler.cc",
    ],
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
        "//band:resource_monitor",
    ],
)

band_cc_library(
    name = "frequency_profiler",
    srcs = [
        "frequency_profiler.cc",
    ],
    hdrs = [
        "frequency_profiler.h",
    ],
    deps = [
        ":profiler",
    ],
)