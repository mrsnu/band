load("//band:band.bzl", "band_cc_android_test")

exports_files(glob([
    "data/*.bin",
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

cc_test(
    name = "planner_test",
    size = "small",
    srcs = ["planner_test.cc"],
    deps = [
        ":test_util",
        "//band:planner",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
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

cc_test(
    name = "worker_test",
    size = "small",
    srcs = ["worker_test.cc"],
    deps = [
        ":test_util",
        "//band:worker",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "time_test",
    size = "small",
    srcs = ["time_test.cc"],
    deps = [
        ":test_util",
        "//band:time",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "config_builder_test",
    srcs = ["config_builder_test.cc"],
    data = ["//band/test:data/config.json"],
    deps = [
        ":test_util",
        "//band:config_builder",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "cpu_test",
    srcs = ["cpu_test.cc"],
    data = [],
    linkopts = ["-ldl"],
    deps = [
        "//band:cpu",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_minimal_test",
    srcs = ["backend/tfl_minimal_test.cc"],
    data = [
        "//band/test:data/add.bin",
        "//band/test:data/config.json",
        "//band/test:data/config_fixed_device.json",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["c/c_api_test.cc"],
    data = [
        "//band/test:data/add.bin",
        "//band/test:data/config.json",
        "//band/test:data/config_fixed_device.json",
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
