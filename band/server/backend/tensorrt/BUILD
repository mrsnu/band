load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

band_cc_library(
    name = "trt_loader",
    hdrs = ["trt_loader.h"],
    deps = [
        "//band/server/util:dso_loader",
        "//band:common",
        "@com_nvidia_tensorrt//:tensorrt",
    ]
)

band_cc_library(
    name = "cuda_loader",
    hdrs = ["cuda_loader.h"],
    deps = [
        "//band/server/util:dso_loader",
        "//band:common",
        "@com_nvidia_cuda//:cuda",
    ]
)

band_cc_library(
    name = "backend",
    hdrs = ["backend.h"],
    srcs = [
        "backend.cc",
    ],
    deps = [
        ":loader",
    ],
)