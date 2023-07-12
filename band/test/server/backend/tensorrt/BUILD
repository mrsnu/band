load("//band:band.bzl", "band_cc_android_test")

band_cc_android_test(
    name = "loader_test",
    size = "small",
    srcs = ["loader_test.cc"],
    deps = [
        "//band/server/backend/tensorrt:trt_loader",
        "@com_google_googletest//:gtest",
    ],
)