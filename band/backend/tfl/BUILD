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
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

band_cc_library(
    name = "tfl_backend",
    srcs = [
        "backend.cc",
        "model.cc",
        "model_executor.cc",
        "tensor.cc",
        "util.cc",
    ],
    hdrs = [
        "backend.h",
        "model.h",
        "model_executor.h",
        "tensor.h",
        "util.h",
    ],
    copts = [
        "-DTF_LITE_STRIP_ERROR_STRINGS",
    ],
    data = [
        "LICENSE"]
         + select({ 
        "//band:android": ["//third_party/hexagon_nn"],
        "//band:android_arm64": ["//third_party/hexagon_nn"],
        "//conditions:default": [],
    }),
    deps = [
        "//band:framework",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:c_api",
        "@org_tensorflow//tensorflow/lite/c:c_api_experimental",
    ] + select({
        "//band:android": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "@org_tensorflow//tensorflow/lite/delegates/hexagon:hexagon_delegate"
        ],
        "//band:android_arm64": [
            "@org_tensorflow//tensorflow/lite/delegates/gpu:delegate",
            "@org_tensorflow//tensorflow/lite/delegates/hexagon:hexagon_delegate"
        ],
        "//conditions:default": [],
    }),
    # force to link band::TfLiteRegisterCreators (in backend.cc)
    # to override the weak symbol in //band:backend_factory
    alwayslink = True,
)
