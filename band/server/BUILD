load("//band:band.bzl", "band_cc_binary", "band_cc_library")

band_cc_library(
    name = "grpc_server",
    srcs = ["grpc_server.cc"],
    hdrs = ["grpc_server.h"],
    deps = [
        "//band/backend/grpc/proto:band_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
    ]
)

band_cc_binary(
    name = "band_server",
    srcs = ["main.cc"],
    deps = [
      ":grpc_server",
    ]
)
