load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

band_cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    deps = [
        "//band/server/util:dso_loader",
        "//band:common",
        "@com_nvidia_tensorrt//:tensorrt",
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