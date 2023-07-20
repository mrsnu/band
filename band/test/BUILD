load("//band:band.bzl", "band_cc_android_test", "band_cc_library")

exports_files(glob([
    "data/*.tflite",
    "data/*.json",
    "data/*.jpg",
]))

band_cc_library(
    name = "test_util",
    srcs = [
        "image_util.cc",
    ],
    hdrs = [
        "image_util.h",
        "test_util.h",
    ],
    copts = [
        "-DSTB_IMAGE_IMPLEMENTATION",
        "-DSTB_IMAGE_WRITE_IMPLEMENTATION",
    ],
    deps = [
        "//band:common",
        "//band/buffer",
        "@com_google_googletest//:gtest",
        "@stblib//:stb_image",
        "@stblib//:stb_image_write",
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
    size = "medium",
    srcs = ["tool/benchmark_test.cc"],
    data = [
        "//band/test:data/add.tflite",
        "//band/test:data/benchmark_config.json",
        "//band/test:data/retinaface_mbv2_quant_160.tflite",
    ],
    deps = [
        ":test_util",
        "//band/tool:benchmark",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "config_builder_test",
    size = "small",
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
        "//band:worker",
        "//band/estimator:latency_estimator",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "cpu_test",
    size = "small",
    srcs = ["cpu_test.cc"],
    data = [],
    deps = [
        "//band/device",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "resource_monitor_test",
    size = "small",
    srcs = ["resource_monitor_test.cc"],
    data = [],
    deps = [
        "//band:resource_monitor",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_minimal_test",
    size = "small",
    srcs = ["backend/tfl_minimal_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:data/add.tflite",
        "//band/test:data/cat.jpg",
        "//band/test:data/lite-model_mobilenet_v2_100_224_fp32_1.tflite",
        "//band/test:data/mobilenet_v2_1.0_224_quant.tflite",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/tfl:tfl_backend",
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "image_operation_test",
    size = "small",
    srcs = ["image_operation_test.cc"],
    data = [
        "//band/test:data/hippo.jpg",
        "//band/test:data/hippo_crop_256.jpg",
        "//band/test:data/hippo_resize_256.jpg",
    ],
    deps = [
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ],
)

band_cc_android_test(
    name = "tfl_subgraph_test",
    size = "enormous",
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
        "//band/test:data/cat.jpg",
        "//band/test:data/lite-model_mobilenet_v2_100_224_fp32_1.tflite",
        "//band/test:data/mobilenet_v2_1.0_224_quant.tflite",
    ],
    deps = [
        "//band:framework",
        "//band/c:c_api",
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)

band_cc_android_test(
    name = "graph_test",
    size = "small",
    srcs = ["graph/graph_test.cc"],
    data = [
        "//band/test:data/add.tflite",
    ],
    deps = [
        "//band:framework",
        "//band/graph:graph",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)