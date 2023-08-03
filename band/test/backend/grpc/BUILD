load("//band:band.bzl", "band_cc_android_test")

band_cc_android_test(
    name = "grpc_client_test",
    size = "small",
    srcs = ["grpc_client_test.cc"],
    data = [
        "//band/test:data/test.band",
    ],
    deps = [
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:grpc": [
            "//band/backend/grpc:grpc_backend",
        ],
        "//conditions:default": [],
    }),
)

band_cc_android_test(
    name = "grpc_minimal_test",
    size = "small",
    srcs = ["grpc_minimal_test.cc"],
    data = [
        "//band/test:data/test.band",
    ],
    deps = [
        "//band:config_builder",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:grpc": [
            "//band/backend/grpc:grpc_backend",
        ],
        "//conditions:default": [],
    }),
)
