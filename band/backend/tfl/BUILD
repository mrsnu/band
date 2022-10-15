load("//band:band.bzl", "band_cc_android_test")
load("//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tfl_backend",
    srcs = [
        "backend.cc",
        "interpreter.cc",
        "model.cc",
        "tensor.cc",
        "util.cc",
    ],
    hdrs = [
        "backend.h",
        "interpreter.h",
        "model.h",
        "tensor.h",
        "util.h",
    ],
    deps = [
        "//band:framework",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api",
        "//tensorflow/lite/c:c_api_experimental",
    ] + select({
        "//band:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//band:android_arm64": [
            "//tensorflow/lite/delegates/gpu:delegate",
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)

band_cc_android_test(
    name = "tfl_minimal_test",
    srcs = ["test/tfl_minimal_test.cc"],
    data = [
        "//band:testdata/add.bin",
        "//band:testdata/config.json",
    ],
    deps = [
        ":tfl_backend",
        "//band:config_builder",
        "@com_google_googletest//:gtest",
    ],
)
