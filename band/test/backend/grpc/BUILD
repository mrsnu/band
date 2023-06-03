load("//band:band.bzl", "band_cc_android_test")

band_cc_android_test(
    name = "grpc_minimal_test",
    size = "small",
    srcs = ["grpc_minimal_test.cc"],
    data = [
        "//band/test:data/test.band",
    ],
    deps = [
        "//band:config_builder",
        "//band/backend/grpc:grpc_backend",
        "@com_google_googletest//:gtest",
    ],
)
