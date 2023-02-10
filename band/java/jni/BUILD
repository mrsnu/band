package(
    default_visibility = ["//band:__subpackages__"],
)

cc_library(
    name = "jni",
    hdrs = select({
        "//band:android": [],
        "//conditions:default": [
            ":jni.h",
            ":jni_md.h",
        ],
    }),
    deps = [
        "//band:common",
        "//band:interface",
        "//band:framework",
        "//band:config_builder",
    ],
    includes = select({
        "//band:android": [],
        "//conditions:default": ["."],
    }),
    visibility = ["//visibility:public"],
)

genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = select({
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)
