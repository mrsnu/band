package(
    default_visibility = ["//band:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

load(
    "//band:band.bzl", 
    "clean_dep",
    "band_cc_library",
    "band_cc_shared_object",
)

band_cc_library(
    name = "tensor",
    srcs = [
        "tensor.cc",
        "buffer.cc",
        "buffer_processor.cc",
        "operation.cc"
    ],
    hdrs = [
        "tensor.h",
        "buffer.h",
        "buffer_processor.h",
        "operation.h",
    ],
    deps = [
        "//band:common",
        "//band:interface",
        "@libyuv",
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
        "//band:common",
        "//band:interface",
        ":tensor"
    ],
)