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
        "jni_utils.cc",
        "band_jni.cc",
        "config_jni.cc",
        "nativeconfigbuilderwrapper_jni.cc",
        "nativeenginewrapper_jni.cc",
        "nativemodelwrapper_jni.cc",
        "nativetensorwrapper_jni.cc",
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