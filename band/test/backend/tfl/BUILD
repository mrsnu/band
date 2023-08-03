load("//band:band.bzl", "band_cc_android_test")

band_cc_android_test(
    name = "tfl_minimal_test",
    size = "medium",
    srcs = ["tfl_minimal_test.cc"],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "//band/test:data/cat.jpg",
        "//band/test:data/add.tflite",
        "//band/test:data/mobilenet_v2_1.0_224_quant.tflite",
        "//band/test:data/lite-model_mobilenet_v2_100_224_fp32_1.tflite",
    ],
    deps = [
        "//band:config_builder",
        "//band/test:test_util",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [],
    }),
)

band_cc_android_test(
    name = "tfl_subgraph_test",
    size = "enormous",
    srcs = ["tfl_subgraph_test.cc"],
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
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [],
    }),
)
