load("//band:band.bzl", "band_copts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "native_framework_only",
    srcs = [
        ":native_framework_only_srcs",
    ],
    hdrs = [":native_hdrs"],
    deps = [
        "//band/java/jni",
    ],
    alwayslink = 1,
)

filegroup(
    name = "native_framework_only_srcs",
    srcs = [
        "nativeenginewrapper_jni.cc",
    ]
)

filegroup(
    name = "native_hdrs",
    srcs = [
        "jni_utils.h",
    ],
)

filegroup(
    name = "native_srcs",
    srcs = [
        "nativeenginewrapper_jni.cc",
    ],
)

filegroup(
    name = "native_api_srcs",
    srcs = [
        "band_jni.cc",
    ],
)

cc_library(
    name = "native",
    srcs = [":native_srcs"],
    hdrs = [":native_hdrs"],
    copts = band_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        "//band/java/jni",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "exported_symbols.lds",
        "version_scripts.lds",
    ]
)