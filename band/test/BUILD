load("//band:band.bzl", "band_cc_android_test")

exports_files(glob([
    "data/*.tflite",
    "data/*.json",
]))

cc_library(
    name = "test_util",
    hdrs = ["test_util.h"],
    deps = [
        "//band:common",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "planner_test",
    size = "small",
    srcs = ["planner_test.cc"],
    deps = [
        ":test_util",
        "//band:planner",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["scheduler_test.cc"],
    deps = [
        ":test_util",
        "//band:config",
        "//band:model",
        "//band:scheduler",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "worker_test",
    size = "small",
    srcs = ["worker_test.cc"],
    deps = [
        ":test_util",
        "//band:worker",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "time_test",
    size = "small",
    srcs = ["time_test.cc"],
    deps = [
        ":test_util",
        "//band:time",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "benchmark_test",
    srcs = ["tool/benchmark_test.cc"],
    data = [
        "//band/test:data/add.tflite",
        "//band/test:data/benchmark_config.json",
    ],
    deps = [
        ":test_util",
        "//band/tool:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "config_builder_test",
    srcs = ["config_builder_test.cc"],
    deps = [
        ":test_util",
        "//band:config_builder",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "latency_estimator_test",
    size = "small",
    srcs = ["latency_estimator_test.cc"],
    deps = [
        ":test_util",
        "//band:config_builder",
        "//band:latency_estimator",
        "//band:worker",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "cpu_test",
    srcs = ["cpu_test.cc"],
    data = [],
    deps = [
        "//band:cpu",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_minimal_test",
    srcs = ["backend/tfl_minimal_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:data/add.tflite",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_subgraph_test",
    srcs = ["backend/tfl_subgraph_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:data/ICN_quant.tflite",
        "//band/test:data/lite-model_efficientdet_lite0_int8_1.tflite",
        "//band/test:data/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite",
        "//band/test:data/retinaface_mbv2_quant_160.tflite",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c/c_api_test.cc"],
    data = [
        "//band/test:data/add.tflite",
    ],
    deps = [
        "//band:framework",
        "//band/c:c_api",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)
