load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

band_cc_library(
    name = "dso_loader",
    srcs = ["dso_loader.cc"],
    hdrs = ["dso_loader.h"],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
    ]
)