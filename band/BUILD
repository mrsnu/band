load("//band:band.bzl", "band_copts")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

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
        "tflite": "true",
    },
    visibility = ["//visibility:public"],
)

# MNN backend setting.
config_setting(
    name = "mnn",
    define_values = {
        "mnn": "true",
    },
    visibility = ["//visibility:public"],
)

# Android config settings
config_setting(
    name = "android",
    flag_values = {},
    values = {
        "crosstool_top": "//external:android/crosstool",
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
    define_values = {
        "tflite": "true",
    },
    flag_values = {},
    values = dict(
        {"crosstool_top": "//external:android/crosstool"},
        cpu = "arm64-v8a",
    ),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm64_mnn",
    define_values = {
        "mnn": "true",
    },
    flag_values = {},
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
        "common.cc",
        "error_reporter.cc",
        "logger.cc",
        "model_spec.cc",
    ],
    hdrs = [
        "common.h",
        "config.h",
        "context.h",
        "cpu.h",
        "error_reporter.h",
        "logger.h",
        "model_spec.h",
    ],
    copts = band_copts(),
    linkopts = select({
        "//band:android": [
            "-llog",
        ],
        "//conditions:default": [
        ],
    }),
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
        "interface/model.h",
        "interface/model_executor.h",
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
    name = "model_analyzer",
    srcs = [
        "model_analyzer.cc",
    ],
    hdrs = [
        "model_analyzer.h",
    ],
    deps = [
        ":backend_factory",
        ":common",
        ":interface",
        ":model",
        ":worker",
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
        "scheduler/fixed_worker_global_queue_scheduler.cc",
        "scheduler/fixed_worker_scheduler.cc",
        "scheduler/heterogeneous_earliest_finish_time_reserved_scheduler.cc",
        "scheduler/least_slack_first_scheduler.cc",
        "scheduler/round_robin_scheduler.cc",
        "scheduler/shortest_expected_latency_scheduler.cc",
    ],
    hdrs = [
        "scheduler/fixed_worker_scheduler.h",
        "scheduler/heterogeneous_earliest_finish_time_reserved_scheduler.h",
        "scheduler/least_slack_first_scheduler.h",
        "scheduler/round_robin_scheduler.h",
        "scheduler/scheduler.h",
        "scheduler/shortest_expected_latency_scheduler.h",
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
    name = "latency_estimator",
    srcs = [
        "latency_estimator.cc",
        "profiler.cc",
    ],
    hdrs = [
        "latency_estimator.h",
        "profiler.h",
    ],
    deps = [
        ":common",
        ":config",
        ":cpu",
        ":json_util",
        ":worker",
    ],
)

cc_library(
    name = "framework",
    srcs = [
        "engine.cc",
        "tensor_ring_buffer.cc",
    ],
    hdrs = [
        "engine.h",
        "tensor_ring_buffer.h",
    ],
    deps =
        [
            ":backend_factory",
            ":common",
            ":config",
            ":interface",
            ":json_util",
            ":latency_estimator",
            ":model",
            ":model_analyzer",
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
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_library(
    name = "config_builder",
    srcs = ["config_builder.cc"],
    hdrs = ["config_builder.h"],
    deps = [
        ":common",
        ":config",
        ":cpu",
        ":macros",
    ],
)
