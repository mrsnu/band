load(
    "//band:band.bzl",
    "band_cc_library",
    "clean_dep",
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
        "benchmark_graph.cc",
        "benchmark_profiler.cc",
        "engine_runner.cc",
        "graph_runner.cc",
        "runner.cc",
    ],
    hdrs = [
        "benchmark.h",
        "benchmark_graph.h",
        "benchmark_profiler.h",
        "engine_runner.h",
        "graph_runner.h",
        "runner.h",
    ],
    deps = [
        "//band:config_builder",
        "//band:framework",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)
