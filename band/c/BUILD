load(
    "//band:band.bzl",
    "band_cc_library",
    "band_cc_shared_object",
    "clean_dep",
)

load(
    "@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar"
)


package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_shared_object(
    name = "band_c",
    linkopts = select({
        "//band:windows": [],
        "//conditions:default": [
            "-z defs",
            "-Wl,--version-script,$(location //band/c:version_scripts.lds)",
        ],
    }),
    per_os_targets = True,
    deps = [
        ":c_api",
        ":version_scripts.lds",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)

cc_binary(
    name = "band_c_main",
    srcs = [
        "example/band_c_main.c",
    ],
    linkopts = select({
        clean_dep("//band:android"): [],
        clean_dep("//band:windows"): [],
        "//conditions:default": [
            "-lpthread",
        ],
    }),
    deps = [
        ":c_api",
    ] + select({
        clean_dep("//band:tflite"): [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [
        ],
    }),
)

band_cc_library(
    name = "c_api",
    srcs = [
        "c_api.cc",
        "c_api_buffer.cc"
    ],
    hdrs = [
        "c_api.h",
        "c_api_buffer.h"
    ],
    deps = [
        ":c_api_internal",
        ":c_api_type",
        "//band:config_builder",
        "//band:framework",
    ],
    alwayslink = 1,
)

band_cc_library(
    name = "c_api_internal",
    srcs = ["c_api_internal.cc"],
    hdrs = ["c_api_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":c_api_type",
        "//band:config_builder",
        "//band:framework",
    ],
)

band_cc_library(
    name = "c_api_type",
    hdrs = ["c_api_type.h"],
)


filegroup(
    name = "c_api_headers",
    srcs = [
        "c_api.h",
        "c_api_buffer.h",
        "c_api_type.h",
        "libband.h",
    ],
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "c_api_headers_pkg",
    srcs = [":c_api_headers"],
    package_dir = "include",
)

pkg_tar(
    name = "c_api_lib_pkg",
    srcs = [":band_c"],
    package_dir = "lib",
)

pkg_tar(
    name = "band_c_pkg",
    deps = [
        ":c_api_headers_pkg",
        ":c_api_lib_pkg",
    ],
)
