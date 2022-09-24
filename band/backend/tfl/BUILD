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
    ],
)

cc_test(
    name = "tfl_minimal_test",
    size = "small",
    srcs = ["test/tfl_minimal_test.cc"],
    data = [
        "//band:testdata/add.bin",
        "//band:testdata/config.json",
    ],
    deps = [
        ":tfl_backend",
        "@com_google_googletest//:gtest",
    ],
)
