load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

proto_library(
    name = "band_proto",
    srcs = [
        "common.proto",
        "request.proto",
        "response.proto",
        "service.proto",
        "tensor.proto",
        "model.proto",
    ],
)

cc_proto_library(
    name = "band_cc_proto",
    deps = [":band_proto"],
)

cc_grpc_library(
    name = "band_cc_grpc",
    srcs = [":band_proto"],
    deps = [":band_cc_proto"],
    grpc_only = True,
)
