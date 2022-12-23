load(":build_defs.bzl", "JAVACOPTS")
load("//band/java:aar_with_jni.bzl", "aar_with_jni", "aar_without_jni")
load("//band:band.bzl", "band_jni_linkopts", "band_jni_binary")
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

# TODO(widiba03304): fill java API sources.
JAVA_API_SRCS = [
    "src/main/java/org/mrsnu/band/BackendType.java",
    "src/main/java/org/mrsnu/band/Config.java",
    "src/main/java/org/mrsnu/band/ConfigBuilder.java",
    "src/main/java/org/mrsnu/band/CpuMaskFlags.java",
    "src/main/java/org/mrsnu/band/DataType.java",
    "src/main/java/org/mrsnu/band/Device.java",
    "src/main/java/org/mrsnu/band/DeviceWorkerAffinityPair.java",
    "src/main/java/org/mrsnu/band/Engine.java",
    "src/main/java/org/mrsnu/band/Model.java",
    "src/main/java/org/mrsnu/band/Quantization.java",
    "src/main/java/org/mrsnu/band/Request.java",
    "src/main/java/org/mrsnu/band/SchedulerType.java",
    "src/main/java/org/mrsnu/band/SubgraphPreparationType.java",
    "src/main/java/org/mrsnu/band/Tensor.java",
    "src/main/java/org/mrsnu/band/WorkerType.java",
]

# Java source files for API uses.
filegroup(
    name = "java_api_srcs",
    srcs = JAVA_API_SRCS,
)

# Java sources excluding API files.
filegroup(
    name = "java_runtime_srcs",
    srcs = [x for x in JAVA_SRCS if x not in JAVA_API_SRCS],
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
    srcs = [":java_runtime_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    exports = [
        ":band_api",
    ],
    deps = [
        ":band_native",
        ":band_api",
        "@org_checkerframework_qual", # Java checker framework
    ]
)

android_library(
    name = "band_api",
    srcs = [":java_api_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "band_java",
    srcs = [":java_srcs"],
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ]
)

java_library(
    name = "band_javalib",
    srcs = [":java_runtime_srcs"],
    javacopts = JAVACOPTS,
    exports = [
        ":band_api_javalib",
    ],
    deps = [
        ":band_api_javalib",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "band_api_javalib",
    srcs = [":java_api_srcs"],
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "band_lib",
    srcs = [":java_runtime_srcs"],
    javacopts = JAVACOPTS,
    exports = [":band_api_javalib"],
    deps = [
        ":band_jni",
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
        "//band/java/src/main/native",
        "//band/c:c_api",
    ],
)
