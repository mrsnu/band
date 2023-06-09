load(
    "//band:band.bzl",
    "band_cc_library",
)

package(
    default_visibility = ["//band:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "buffer",
    srcs = [
        "buffer.cc",
        "image_operation.cc",
        "image_processor.cc",
        "libyuv_operation.cc",
        "operation.cc",
        "processor.cc",
    ],
    hdrs = [
        "buffer.h",
        "image_operation.h",
        "image_processor.h",
        "libyuv_operation.h",
        "operation.h",
        "processor.h",
    ],
    deps = [
        "//band:common",
        "//band:interface",
        "@libyuv",
    ],
)
