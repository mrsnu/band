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

load("//band:band.bzl", "band_cc_library")

package(
    default_visibility = ["//band:__subpackages__"],
)

band_cc_library(
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
