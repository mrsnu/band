load("//band:band.bzl", "band_copts", "band_cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
        "band_jni.cc",
        "config_jni.cc",
        "image_processor_jni.cc",
        "jni_utils.cc",
        "native_buffer_wrapper_jni.cc",
        "native_config_builder_wrapper_jni.cc",
        "native_engine_wrapper_jni.cc",
        "native_image_processor_builder_wrapper_jni.cc",
        "native_model_wrapper_jni.cc",
        "native_tensor_wrapper_jni.cc",
    ],
)

band_cc_library(
    name = "native",
    srcs = [":native_srcs"],
    hdrs = [":native_hdrs"],
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