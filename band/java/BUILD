load(":build_defs.bzl")
load("//band/java:aar_with_jni.bzl", "aar_with_jni")
load("//band:band.bzl", "clean_dep", "band_jni_linkopts", "band_jni_binary")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

JAVA_SRCS = glob([
    "src/main/java/org/mrsnu/band/*.java",
])

# All Java source files.
filegroup(
    name = "java_srcs",
    srcs = JAVA_SRCS,
)

# AAR Targets
aar_with_jni(
    name = "band",
    android_library = ":band_runtime",
    headers = [
        "//band/c:c_api.h",
        "//band/c:c_api_types.h",
        "//band/c:common.h",
    ],
)

android_library(
    name = "band_runtime",
    srcs = [":java_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    exports = [
        ":band_api",
    ],
    deps = [
        ":band_native",
        ":band_api",
        "@org_checkerframework_qual", # Java checker framework
    ],
)

android_library(
    name = "band_api",
    srcs = [":java_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

cc_library(
    name = "band_native",
    srcs = [
        ":band_jni"
    ],
)

band_jni_binary(
    name = "band_jni",
    linkopts = band_jni_linkopts() + ["-Wl,--no-undefined"],
    linkscript = ":band_version_script.lds",
    deps = [
        "//band/c:c_api",
        "//band/java/src/main/native",
    ] + select({
        clean_dep("//band:tflite"): [
            "//band/backend/tfl:tfl_backend",
        ],
        "//conditions:default": [],
    }),
)
