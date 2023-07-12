package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "cudnn_headers",
    hdrs = [
        ":cudnn_include",
    ],
    include_prefix = "third_party/cudnn",
    strip_include_prefix = "cudnn/include",
)

cc_library(
    name = "cudnn",
    srcs = [
        ":cudnn_lib",
    ],
    data = [
        ":cudnn_lib",
    ],
    deps = [
        ":cudnn_headers",
    ]
)

COPY_RULES
