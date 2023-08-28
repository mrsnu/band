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

"""Android configuration utility"""

_ANDROID_NDK_HOME = "ANDROID_NDK_HOME"
_ANDROID_SDK_HOME = "ANDROID_SDK_HOME"
_ANDROID_NDK_API_LEVEL = "ANDROID_NDK_API_LEVEL"
_ANDROID_SDK_API_LEVEL = "ANDROID_SDK_API_LEVEL"
_ANDROID_BUILD_TOOLS_VERSION = "ANDROID_BUILD_TOOLS_VERSION"

def _init_android_impl(ctx):
    if all([
        ctx.os.environ.get(_ANDROID_BUILD_TOOLS_VERSION),
        ctx.os.environ.get(_ANDROID_SDK_HOME),
        ctx.os.environ.get(_ANDROID_SDK_API_LEVEL),
    ]):
        native.android_sdk_repository(
            name = "androidsdk",
            path = ctx.environ.get(_ANDROID_SDK_HOME),
            api_level = ctx.environ.get(_ANDROID_SDK_API_LEVEL),
            build_tools_version = ctx.environ.get(_ANDROID_BUILD_TOOLS_VERSION),
        )

    if all([
        ctx.os.environ.get(_ANDROID_NDK_HOME),
        ctx.os.environ.get(_ANDROID_NDK_API_LEVEL),
    ]):
        native.android_ndk_repository(
            name = "androidndk",
            path = ctx.environ.get(_ANDROID_NDK_HOME),
            api_level = ctx.environ.get(_ANDROID_NDK_API_LEVEL),
        )

repo = repository_rule(
    implementation = _init_android_impl,
    environ = [
        _ANDROID_NDK_HOME,
        _ANDROID_SDK_HOME,
        _ANDROID_NDK_API_LEVEL,
        _ANDROID_SDK_API_LEVEL,
        _ANDROID_BUILD_TOOLS_VERSION,
    ],
)