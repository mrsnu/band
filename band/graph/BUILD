load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "graph",
    hdrs = [
        "graph.h",
        "node.h",
        "graph_builder.h",
    ],
    srcs = [
        "graph.cc",
        "node.cc",
        "graph_builder.cc",
    ],
    deps = [
        "//band:model",
        "//band:tensor",
    ]
)