load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

band_cc_library(
    name = "grpc_client",
    hdrs = [
        "grpc_client.h"
    ],
    deps = [
        "//band/backend/grpc/proto:band_cc_grpc",
        "@com_github_grpc_grpc//:grpc++",
    ]
)

band_cc_library(
    name = "grpc_backend",
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
    # force to link band::GrpcRegisterCreators (in backend.cc)
    # to override the weak symbol in //band:backend_factory
    alwayslink = True,
    deps = [
        "//band:framework",
        "//band/backend/grpc:grpc_client",
    ],
)