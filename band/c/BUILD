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
        ":c_api_type",
        "//band:config_builder",
        "//band:common",
        "//band:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_type",
    hdrs = ["c_api_type.h"],
    copts = band_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":common",
        "//band:config_builder",
        "//band:common",
        "//band:framework",
    ],
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
