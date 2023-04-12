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
    alwayslink = True,
    deps = [
        "//band:framework",
    ],
)