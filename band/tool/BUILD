load(
    "//band:band.bzl", 
    "clean_dep",
    "band_cc_library",
)
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "band_benchmark",
    srcs = [
        "band_benchmark.cc",
    ],
    linkopts = select({
        clean_dep("//band:android"): [
            "-pie",
            "-lm",
            "-Wl,--rpath=/data/local/tmp/",
        ],
        "//conditions:default": [],
    }),
    linkstatic = True,
    deps = [
        ":benchmark",
    ],
)


band_cc_library(
    name = "benchmark",
    srcs = [
        "benchmark.cc",
        "benchmark_profiler.cc",
        "benchmark_instance.cc"
    ],
    hdrs = [
        "benchmark.h",
        "benchmark_config.h",
        "benchmark_profiler.h",
        "benchmark_instance.h"
    ],
    deps = [
        "//band:framework",
        "//band:config_builder",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)