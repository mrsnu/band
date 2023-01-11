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
    # force to link Band::TfLiteRegisterCreators (in backend.cc)
    # to override the weak symbol in //band:backend_factory
    alwayslink = True,
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS"
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