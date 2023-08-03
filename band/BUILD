load("//band:band.bzl", "band_cc_library")

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

# GRPC backend setting.
config_setting(
    name = "grpc",
    define_values = {
        "grpc": "true",
    },
    visibility = ["//visibility:public"],
)

# Linux config settings
config_setting(
    name = "linux",
    flag_values = {},
    values = {},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    flag_values = {},
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64_tflite",
    define_values = {
        "tflite": "true",
    },
    flag_values = {},
    values = {"cpu": "k8"},
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

# Windows config settings
config_setting(
    name = "windows",
    flag_values = {},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_tflite",
    define_values = {
        "tflite": "true",
    },
    flag_values = {},
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

# Tracer config setting

config_setting(
    name = "trace",
    define_values = {
        "trace": "true",
    },
    visibility = ["//visibility:public"],
)

band_cc_library(
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
        "engine_interface.h",
        "error_reporter.h",
        "logger.h",
        "model_spec.h",
    ],
    linkopts = select({
        "//band:android": [
            "-llog",
        ],
        "//conditions:default": [
        ],
    }),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

band_cc_library(
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
    deps = [
        ":common",
        "//band/device",
    ],
)

band_cc_library(
    name = "backend_factory",
    srcs = [
        "backend_factory.cc",
    ],
    hdrs = [
        "backend_factory.h",
    ],
    deps = [
        ":common",
        ":interface",
    ],
)

band_cc_library(
    name = "config",
    hdrs = [
        "config.h",
    ],
    linkopts = select(
        {
            "//band:windows": [],
            "//conditions:default": [
                "-lm",
                "-ldl",
            ],
        },
    ),
    deps = [
        ":common",
        "//band/device",
    ],
)

band_cc_library(
    name = "time",
    srcs = [
        "time.cc",
    ],
    hdrs = [
        "time.h",
    ],
)

band_cc_library(
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

band_cc_library(
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

band_cc_library(
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
        ":job_tracer",
        ":time",
        "//band/device",
    ],
)

band_cc_library(
    name = "scheduler",
    srcs = [
        "scheduler/fixed_worker_global_queue_scheduler.cc",
        "scheduler/fixed_worker_scheduler.cc",
        "scheduler/heterogeneous_earliest_finish_time_scheduler.cc",
        "scheduler/least_slack_first_scheduler.cc",
        "scheduler/round_robin_scheduler.cc",
        "scheduler/shortest_expected_latency_scheduler.cc",
    ],
    hdrs = [
        "scheduler/fixed_worker_scheduler.h",
        "scheduler/heterogeneous_earliest_finish_time_scheduler.h",
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

band_cc_library(
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
        ":job_tracer",
        ":scheduler",
        ":worker",
    ],
)

band_cc_library(
    name = "job_tracer",
    srcs = [
        "job_tracer.cc",
    ],
    hdrs = [
        "job_tracer.h",
    ],
    deps = [
        ":common",
        "@chrome_tracer//chrome_tracer",
    ],
)

band_cc_library(
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
        ":json_util",
        ":worker",
        "//band/device",
    ],
)

band_cc_library(
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

band_cc_library(
    name = "tensor_ring_buffer",
    srcs = [
        "tensor_ring_buffer.cc",
    ],
    hdrs = [
        "tensor_ring_buffer.h",
    ],
    deps = [
        ":common",
        ":interface",
        ":tensor",
    ],
)

band_cc_library(
    name = "framework",
    srcs = [
        "engine.cc",
    ],
    hdrs = [
        "engine.h",
    ],
    deps = [
        ":backend_factory",
        ":common",
        ":config",
        ":interface",
        ":job_tracer",
        ":json_util",
        ":latency_estimator",
        ":model",
        ":model_analyzer",
        ":planner",
        ":resource_monitor",
        ":scheduler",
        ":tensor",
        ":tensor_ring_buffer",
        ":time",
        ":worker",
        "//band/buffer",
    ],
)

band_cc_library(
    name = "json_util",
    srcs = ["json_util.cc"],
    hdrs = ["json_util.h"],
    linkopts = select(
        {
            "//band:windows": [],
            "//conditions:default": [
                "-lm",
                "-ldl",
            ],
        },
    ),
    deps = [
        ":common",
        "@jsoncpp",
    ],
)

band_cc_library(
    name = "config_builder",
    srcs = ["config_builder.cc"],
    hdrs = ["config_builder.h"],
    deps = [
        ":common",
        ":config",
        "//band/device",
    ],
)

band_cc_library(
    name = "resource_monitor",
    srcs = ["resource_monitor.cc"],
    hdrs = ["resource_monitor.h"],
    deps = [
        ":common",
        "//band/device",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)
