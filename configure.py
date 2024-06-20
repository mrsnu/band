#!/usr/bin/env python
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

import os
import platform
import argparse

HEXAGON_LICENSE_PATH = "third_party/hexagon_nn/CLICK_THROUGH_LICENSE.txt"


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_cygwin():
    return platform.system().startswith("CYGWIN_NT")


def get_home():
    global env
    if is_windows() or is_cygwin():
        home_path = env["APPDATA"].replace("\\", "/")
    elif is_linux():
        home_path = env["HOME"]
    else:
        raise EnvironmentError("Only Linux or Windows is supported to build.")
    return home_path


def get_var(var, default):
    global env
    if env.get(var) is not None:
        return env.get(var)
    else:
        print(f"{var} is not set. {default} is set by default.")
        return default


def validate_android_tools(sdk_path, ndk_path):
    if not os.path.exists(sdk_path):
        return False
    if not os.path.exists(ndk_path):
        return False
    return True


def validate_android_build_tools(sdk_path, version):
    build_tools_path = os.path.exists(os.path.join(sdk_path, "build-tools"))
    if not build_tools_path:
        return False
    versions = sorted(os.listdir(build_tools_path))
    if version not in versions:
        return False
    return True


def validate_android_api_level(sdk_path, api_level):
    if not os.path.exists(os.path.join(sdk_path, "android-" + api_level)):
        return False
    return True


class BazelConfig(object):
    def __init__(self):
        self.build_configs = {}

    def add_config(self, env_var, value):
        self.build_configs[env_var] = value

    def get_bazel_build_config(self):
        result = ""
        for k, v in self.build_configs.items():
            result += f'build --action_env {k}="{v}"' + "\n"
        return result

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.get_bazel_build_config())


def main():
    global env
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=str,
        default=os.path.abspath(os.path.dirname(__file__)),
        required=False,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".band_android_config.bazelrc",
        required=False,
    )

    args = parser.parse_args()

    ROOT_DIR = args.workspace
    
    # check first if hexagon license is accepted
    hexagon_license_path = os.path.join(ROOT_DIR, HEXAGON_LICENSE_PATH)
    if not os.path.exists(hexagon_license_path):
        raise EnvironmentError(
            "Cannot find hexagon license file. Please check HEXAGON_LICENSE_PATH."
        )
    
    # print content of license file and ask user to accept
    with open(hexagon_license_path, "rb") as f:
        for line in f:
            print(line.decode("ISO-8859-1"), end="")
        print("Do you accept the license? [y/n]")
        accept = input()
        if accept != "y":
            raise EnvironmentError(
                "Hexagon license is not accepted. Please accept the license to crosscompile towards android."
            )


    env = dict(os.environ)

    android_build_tools_version = get_var("ANDROID_BUILD_TOOLS_VERSION", "30.0.0")

    android_sdk_path = get_var(
        "ANDROID_SDK_HOME", os.path.join(get_home(), "Android/Sdk")
    )
    android_ndk_path = get_var(
        "ANDROID_NDK_HOME", os.path.join(get_home(), "Android/Sdk/ndk-bundle")
    )

    if not validate_android_tools(android_sdk_path, android_ndk_path):
        raise EnvironmentError(
            "Cannot find android sdk & ndk. Please check ANDROID_SDK_HOME & ANDROID_NDK_HOME."
        )
    platforms = os.path.join(android_sdk_path, "platforms")
    api_levels = sorted(os.listdir(platforms))
    api_levels = [level.replace("android-", "") for level in api_levels]

    # ANDROID_API_LEVEL
    android_api_level = get_var("ANDROID_API_LEVEL", "28")

    # ANDROID_NDK_API_LEVEL
    android_ndk_api_level = get_var("ANDROID_NDK_API_LEVEL", "21")

    config_file_path = os.path.join(ROOT_DIR, args.output)
    bazel_config = BazelConfig()
    bazel_config.add_config("ANDROID_BUILD_TOOLS_VERSION", android_build_tools_version)
    bazel_config.add_config("ANDROID_SDK_API_LEVEL", android_api_level)
    bazel_config.add_config("ANDROID_SDK_HOME", android_sdk_path)
    bazel_config.add_config("ANDROID_NDK_API_LEVEL", android_ndk_api_level)
    bazel_config.add_config("ANDROID_NDK_HOME", android_ndk_path)
    bazel_config.save(config_file_path)
    print(f"Successfully saved Android configuration to {config_file_path}")

if __name__ == "__main__":
    main()
