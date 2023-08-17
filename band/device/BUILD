load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = [
        "//band:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "macros",
    hdrs = [
        "macros.h",
    ],
)

band_cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        ":macros",
        "//band:common",
        "@com_google_absl//absl/status",
    ],
)

band_cc_library(
    name = "device",
    srcs = [
        "device.cc",
    ],
    hdrs = [
        "device.h",
    ],
    deps = [
        ":util",
    ],
)

band_cc_library(
    name = "cpu",
    srcs = [
        "cpu.cc",
    ],
    hdrs = [
        "cpu.h",
    ],
    deps = [
        ":util",
    ],
)

band_cc_library(
    name = "thermal",
    srcs = [
        "thermal.cc",
    ],
    hdrs = [
        "thermal.h",
    ],
    deps = [
        ":util",
        "//band:common",
        "//band:config",
    ],
)

band_cc_library(
    name = "frequency",
    srcs = [
        "frequency.cc",
    ],
    hdrs = [
        "frequency.h",
    ],
    deps = [
        ":util",
        "//band:common",
        "//band:config",
    ],
)