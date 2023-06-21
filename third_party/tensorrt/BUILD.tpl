package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt_headers",
    hdrs = [
        ":tensorrt_include",
    ],
    include_prefix = "third_party/tensorrt",
    strip_include_prefix = "tensorrt/include",
)

cc_library(
    name = "tensorrt",
    srcs = [
        ":tensorrt_lib",
    ],
    data = [
        ":tensorrt_lib",
    ],
    deps = [
        ":tensorrt_headers",
    ]
)

COPY_RULES
