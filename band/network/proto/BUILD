load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

exports_files([
    "band.proto",
])

proto_library(
    name = "band_proto",
    srcs = ["band.proto"],
)

cc_proto_library(
    name = "band_cc_proto",
    deps = [":band_proto"],
)

cc_grpc_library(
    name = "proto",
    srcs = [":band_proto"],
    grpc_only = True,
    visibility = ["//visibility:public"],
    deps = [":band_cc_proto"],
)
