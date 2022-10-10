load(
    "//band:band.bzl", 
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
        "//tensorflow:windows": [],
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
    }),
)

cc_library(
    name = "bandc_wrapper",
    srcs = [
        "test/bandc_wrapper.cc",
    ],
    hdrs = [
        "test/bandc_wrapper.h",
    ],
    deps = [
        ":c_api",
    ]
)

cc_library(
    name = "common",
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
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
        "//band:common",
        "//band:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_internal",
    hdrs = ["c_api_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":common",
        "//band:framework",
        "//band:common",
        "//band:config_builder",
    ],
)

cc_library(
    name = "c_api_types",
    hdrs = ["c_api_types.h"],
    copts = band_copts(),
)

cc_test(
    name = "c_api_test",
    size = "small",
    srcs = ["test/c_api_test.cc"],
    data = [
        "//band:testdata/add.bin",
        "//band:testdata/config.json",
    ],
    deps = [
        "//band:framework",
        ":c_api",
        "@com_google_googletest//:gtest",
    ] + select({
        "//band:tflite": [
            "//band/backend/tfl:tfl_backend",
        ],
    }),
)
