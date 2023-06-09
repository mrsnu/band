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

band_cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
    ],
    hdrs = [
        "tensor.h",
    ],
    deps = [
        ":buffer",
        "//band:common",
        "//band:interface",
    ],
)

band_cc_library(
    name = "tensor_ring_buffer",
    srcs = [
        "tensor_ring_buffer.cc",
    ],
    hdrs = [
        "tensor_ring_buffer.h",
    ],
    deps = [
        ":tensor",
        "//band:common",
        "//band:interface",
    ],
)
