# Copyright 2023 Seoul National University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//band/java:aar_with_jni.bzl", "aar_with_jni")
load("//band:band.bzl", "band_cc_library", "band_jni_binary", "band_jni_linkopts", "clean_dep")
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
    external_libraries = select({
        clean_dep("//band:tflite"): [
            "//third_party/hexagon_nn",
        ],
        "//conditions:default": [],
    }),
    headers = [
        "//band/c:c_api.h",
        "//band/c:c_api_type.h",
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
        ":band_api",
        ":band_native",
        "@org_checkerframework_qual",  # Java checker framework
    ],
)

android_library(
    name = "band_api",
    srcs = [
        ":java_srcs",
    ],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

band_cc_library(
    name = "band_native",
    srcs = [
        ":band_jni",
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
