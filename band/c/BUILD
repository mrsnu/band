load(
    "//band:band.bzl",
    "band_cc_library",
    "band_cc_shared_object",
    "clean_dep",
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
    ],
    hdrs = [
        "c_api.h",
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
