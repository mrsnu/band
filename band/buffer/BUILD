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
        "common_operator.cc",
        "image_operator.cc",
        "image_processor.cc",
        "libyuv_image_operator.cc",
        "operator.cc",
    ],
    hdrs = [
        "buffer.h",
        "buffer_processor.h",
        "common_operator.h",
        "image_operator.h",
        "image_processor.h",
        "libyuv_image_operator.h",
        "operator.h",
    ],
    deps = [
        "//band:common",
        "//band:interface",
        "@libyuv",
    ],
)
