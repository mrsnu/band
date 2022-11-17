load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

exports_files([
    "splash.proto",
])

proto_library(
    name = "splash_proto",
    srcs = ["splash.proto"],
)

cc_proto_library(
    name = "splash_cc_proto",
    deps = [":splash_proto"],
)

cc_grpc_library(
    name = "splash_grpc_proto",
    srcs = [":splash_proto"],
    deps = [":splash_cc_proto"],
    grpc_only = True,
    visibility = ["//visibility:public"]
)