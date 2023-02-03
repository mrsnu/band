package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tfl_backend",
    srcs = [
        "backend.cc",
        "model_executor.cc",
        "model.cc",
        "tensor.cc",
        "util.cc",
    ],
    hdrs = [
        "backend.h",
        "model_executor.h",
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
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:c_api",
        "@org_tensorflow//tensorflow/lite/c:c_api_experimental",
    ] + select({
        "//band:android": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "@org_tensorflow//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//band:android_arm64": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "@org_tensorflow//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)