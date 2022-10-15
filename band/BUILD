load("//band:band.bzl", "band_copts")

config_setting(
    name = "debug",
    values = {
        "compilation_mode": "dbg",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "optimized",
    values = {
        "compilation_mode": "opt",
    },
    visibility = ["//visibility:public"],
)

# TFLite backend setting.
config_setting(
    name = "tflite",
    define_values = {
        "tflite": "true"
    },
    visibility = ["//visibility:public"],
)

# MNN backend setting.
config_setting(
    name = "mnn",
    define_values = {
        "mnn": "true"
    },
    visibility = ["//visibility:public"],
)

# Android config settings
config_setting(
    name = "android",
    flag_values = {},
    values = {
        "crosstool_top": "//external:android/crosstool"
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64",
    flag_values = {},
    values = dict(
        {"crosstool_top": "//external:android/crosstool"},
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64_tflite",
    flag_values = {},
    define_values = {
        "tflite": "true",
    },
    values = dict(
        {"crosstool_top": "//external:android/crosstool"},
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64_mnn",
    flag_values = {},
    define_values = {
        "mnn": "true"
    },
    values = dict(
        {"crosstool_top": "//external:android/crosstool"},
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    flag_values = {},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common",
    srcs = [
        "error_reporter.cc",
        "logger.cc",
    ],
    hdrs = [
        "common.h",
        "context.h",
        "error_reporter.h",
        "logger.h",
    ],
    copts = band_copts(),
    visibility = [
        "//band:__subpackages__",
    ],
    deps = [
        "//band/c:common",
    ],
)

cc_library(
    name = "interface",
    srcs = [
        "interface/tensor.cc",
    ],
    hdrs = [
        "interface/backend.h",
        "interface/interpreter.h",
        "interface/model.h",
        "interface/tensor.h",
        "interface/tensor_view.h",
    ],
    copts = band_copts(),
    deps = [
        ":common",
    ],
)

cc_library(
    name = "backend_factory",
    srcs = [
        "backend_factory.cc",
    ],
    hdrs = [
        "backend_factory.h",
    ],
    copts = band_copts(),
    deps = [
        ":common",
        ":interface",
    ],
)

cc_library(
    name = "cpu",
    srcs = [
        "cpu.cc",
    ],
    hdrs = [
        "cpu.h",
    ],
    deps = [
        ":common",
    ],
)

cc_library(
    name = "config",
    hdrs = [
        "config.h",
    ],
    linkopts = select({
        "//band:windows": [],
        "//conditions:default": [
            "-lm",
            "-ldl",
        ],
    }),
    deps = [
        ":common",
        ":cpu",
    ],
)

cc_library(
    name = "time",
    srcs = [
        "time.cc",
    ],
    hdrs = [
        "time.h",
    ],
)

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "model.h",
    ],
    deps = [
        ":backend_factory",
        ":interface",
    ],
)

cc_library(
    name = "worker",
    srcs = [
        "worker.cc",
        "worker_device_queue.cc",
        "worker_global_queue.cc",
    ],
    hdrs = [
        "worker.h",
    ],
    deps = [
        ":common",
        ":config",
        ":cpu",
        ":time",
    ],
)

cc_library(
    name = "scheduler",
    srcs = [
        "scheduler/fixed_device_global_queue_scheduler.cc",
        "scheduler/fixed_device_scheduler.cc",
        "scheduler/round_robin_scheduler.cc",
    ],
    hdrs = [
        "scheduler/fixed_device_scheduler.h",
        "scheduler/round_robin_scheduler.h",
        "scheduler/scheduler.h",
    ],
    deps = [
        ":common",
        ":time",
    ],
)

cc_library(
    name = "planner",
    srcs = [
        "planner.cc",
        "safe_bool.cc",
    ],
    hdrs = [
        "planner.h",
        "safe_bool.h",
    ],
    deps = [
        ":common",
        ":scheduler",
        ":worker",
    ],
)

cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = [
        "tensor.h",
    ],
    deps = [
        ":common",
        ":interface",
    ],
)

cc_library(
    name = "framework",
    srcs = [
        "engine.cc",
        "profiler.cc",
        "tensor_ring_buffer.cc",
    ],
    hdrs = [
        "engine.h",
        "profiler.h",
        "tensor_ring_buffer.h",
    ],
    visibility = [
        "//band:__subpackages__",
    ],
    deps =
        [
            ":backend_factory",
            ":common",
            ":config",
            ":json_util",
            ":interface",
            ":model",
            ":planner",
            ":scheduler",
            ":tensor",
            ":time",
            ":worker",
        ],
)

cc_library(
    name = "json_util",
    srcs = ["json_util.cc"],
    hdrs = ["json_util.h"],
    linkopts = select({
        "//tensorflow:windows": [],
        "//conditions:default": [
            "-lm",
            "-ldl",
        ],
    }),
    deps = [
        ":common",
        "@jsoncpp_git//:jsoncpp",
    ]
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "config_builder",
    srcs = ["config_builder.cc"],
    hdrs = ["config_builder.h"],
    visibility = [
        "//band:__subpackages__",
    ],
    deps = [
        ":common",
        ":cpu",
        ":config",
        ":macros"
    ],
)

exports_files(glob([
    "testdata/*.bin",
    "testdata/*.json",
]))

cc_library(
    name = "test_util",
    hdrs = ["test/test_util.h"],
    deps = [
        ":common",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "planner_test",
    size = "small",
    srcs = ["test/planner_test.cc"],
    deps = [
        ":test_util",
        ":planner",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "scheduler_test",
    size = "small",
    srcs = ["test/scheduler_test.cc"],
    deps = [
        ":test_util",
        ":scheduler",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "worker_test",
    size = "small",
    srcs = ["test/worker_test.cc"],
    deps = [
        ":test_util",
        ":worker",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "time_test",
    size = "small",
    srcs = ["test/time_test.cc"],
    deps = [
        ":test_util",
        ":time",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "config_builder_test",
    srcs = ["config_builder_test.cc"],
    data = ["//band:testdata/config.json"],
    deps = [
        ":test_util",
        ":config_builder",
        "@com_google_googletest//:gtest",
    ],
)