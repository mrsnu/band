load("//band:band.bzl", "band_cc_library", "clean_dep")

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
        "//band:json_util",
        "//band:worker",
        "//band/device",
        "//band/profiler:latency_profiler",
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
        "//band:json_util",
        "//band:worker",
        "//band/device",
        "//band/profiler:latency_profiler",
        "//band/profiler:thermal_profiler",
        "//third_party/eigen3",
    ],
)

band_cc_library(
    name = "estimators",
    deps = [":latency_estimator"] + select({
        clean_dep("//band:splash"): [
            ":thermal_estimator",
        ],
        "//conditions:default": [],
    }),
)
