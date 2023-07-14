package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "device",
    srcs = [
        "cpu.cc",
        "util.cc",
    ],
    hdrs = [
        "cpu.h",
        "util.h",
    ],
    deps = [
        "//band:common",
        "@com_google_absl//absl/status",
    ],
)
