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
        "buffer_processor.cc",
        "image_operation.cc",
        "image_processor.cc",
        "libyuv_operation.cc",
        "operation.cc",
    ],
    hdrs = [
        "buffer.h",
        "buffer_processor.h",
        "image_operation.h",
        "image_processor.h",
        "libyuv_operation.h",
        "operation.h",
    ],
    deps = [
        "//band:common",
        "//band:interface",
        "@libyuv",
    ],
)
