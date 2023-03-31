load(
    "//band:band.bzl", 
    "clean_dep",
    "band_cc_shared_object",
    "band_copts",
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

cc_library(
    name = "c_api",
    srcs = [
        "c_api.cc",
    ],
    hdrs = [
        "c_api.h",
    ],
    copts = band_copts(),
    deps = [
        ":c_api_types",
        ":c_api_internal",
        "//band:config_builder",
        "//band:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    srcs = ["c_api_internal.cc"],
    visibility = ["//visibility:private"],
    deps = [
        ":c_api_types",
        "//band:framework",
        "//band:config_builder",
    ],
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    copts = band_copts(),
)