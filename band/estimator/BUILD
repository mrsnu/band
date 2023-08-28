load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "estimator_interface",
    hdrs = [
        "estimator_interface.h",
    ],
    deps = [
        "//band:common",
        "//band:config",
    ],
)

band_cc_library(
    name = "latency_estimator",
    srcs = [
        "latency_estimator.cc",
    ],
    hdrs = [
        "latency_estimator.h",
    ],
    deps = [
        ":estimator_interface",
        "//band/profiler:latency_profiler",
        "//band:json_util",
        "//band:worker",
        "//band/device",
    ],
)

band_cc_library(
    name = "thermal_estimator",
    srcs = [
        "thermal_estimator.cc",
    ],
    hdrs = [
        "thermal_estimator.h",
    ],
    deps = [
        ":estimator_interface",
        "//band/profiler:thermal_profiler",
        "//band/profiler:frequency_profiler",
        "//band/profiler:latency_profiler",
        "//band:json_util",
        "//band:worker",
        "//band/device",
        "//third_party/eigen3"
    ] + select({
        "//conditions:default": [
            "//band/estimator:latency_estimator",
        ],
        "//band:freq": [
            "//band/estimator:frequency_latency_estimator",
        ],
    }),
)

band_cc_library(
    name = "frequency_latency_estimator",
    srcs = [
        "frequency_latency_estimator.cc",
    ],
    hdrs = [
        "frequency_latency_estimator.h",
    ],
    deps = [
        ":estimator_interface",
        "//band/profiler:latency_profiler",
        "//band/profiler:frequency_profiler",
        "//band:json_util",
        "//band:worker",
        "//band/device",
    ],
)

band_cc_library(
    name = "estimators",
    deps = [
        ":latency_estimator",
        ":thermal_estimator",
        ":frequency_latency_estimator",
    ],
)