#!/usr/bin/env python
import os
import platform
import argparse


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_cygwin():
    return platform.system().startswith("CYGWIN_NT")


def get_home():
    global env
    if is_windows() or is_cygwin():
        home_path = env['APPDATA'].replace('\\', '/')
    elif is_linux():
        home_path = env['HOME']
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
    build_tools_path = os.path.exists(os.path.join(sdk_path, 'build-tools'))
    if not build_tools_path:
        return False
    versions = sorted(os.listdir(build_tools_path))
    if version not in versions:
        return False
    return True


def validate_android_api_level(sdk_path, api_level):
    if not os.path.exists(os.path.join(sdk_path, 'android-' + api_level)):
        return False
    return True


def validate_tensorrt(tensorrt_include_path, tensorrt_lib_path):
    if not os.path.exists(tensorrt_include_path):
        return os.path.exists(os.path.join(tensorrt_include_path, "NvInfer.h"))
    if not os.path.exists(tensorrt_lib_path):
        return os.path.exists(os.path.join(tensorrt_lib_path, "libnvinfer.so"))
    return True


class BazelConfig(object):
    def __init__(self):
        self.build_configs = {}

    def add_config(self, env_var, value):
        self.build_configs[env_var] = value

    def get_bazel_build_config(self):
        result = ""
        for k, v in self.build_configs.items():
            result += f"build --action_env {k}=\"{v}\"" + "\n"
        return result

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.get_bazel_build_config())


def main():
    global env
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str,
                        default=os.path.abspath(os.path.dirname(__file__)), required=False)
    args = parser.parse_args()

    ROOT_DIR = args.workspace

    env = dict(os.environ)

    android_build_tools_version = get_var(
        "ANDROID_BUILD_TOOLS_VERSION", "30.0.0")

    android_sdk_path = get_var(
        "ANDROID_SDK_HOME", os.path.join(get_home(), "Android/Sdk"))
    android_ndk_path = get_var("ANDROID_NDK_HOME", os.path.join(
        get_home(), "Android/Sdk/ndk-bundle"))

    if not validate_android_tools(android_sdk_path, android_ndk_path):
        raise EnvironmentError(
            "Cannot find android sdk & ndk. Please check ANDROID_SDK_HOME & ANDROID_NDK_HOME.")
    platforms = os.path.join(android_sdk_path, 'platforms')
    api_levels = sorted(os.listdir(platforms))
    api_levels = [level.replace('android-', '') for level in api_levels]

    # ANDROID_API_LEVEL
    android_api_level = get_var("ANDROID_API_LEVEL", "28")

    # ANDROID_NDK_API_LEVEL
    android_ndk_api_level = get_var("ANDROID_NDK_API_LEVEL", "21")
    
    config_file_path = os.path.join(ROOT_DIR, ".band_android_config.bazelrc")
    android_bazel_config = BazelConfig()
    android_bazel_config.add_config("ANDROID_BUILD_TOOLS_VERSION", android_build_tools_version)
    android_bazel_config.add_config("ANDROID_SDK_API_LEVEL", android_api_level)
    android_bazel_config.add_config("ANDROID_SDK_HOME", android_sdk_path)
    android_bazel_config.add_config("ANDROID_NDK_API_LEVEL", android_ndk_api_level)
    android_bazel_config.add_config("ANDROID_NDK_HOME", android_ndk_path)
    android_bazel_config.save(config_file_path)
    print(f"Successfully saved Android configuration to {config_file_path}")

    # TensorRT include & lib path
    tensorrt_include_path = get_var(
        "TENSORRT_INCLUDE_PATH", "/usr/include/x86_64-linux-gnu")
    tensorrt_lib_path = get_var(
        "TENSORRT_LIB_PATH", "/usr/lib/x86_64-linux-gnu")
    config_file_path = os.path.join(ROOT_DIR, ".band_tensorrt_config.bazelrc")
    tensorrt_bazel_config = BazelConfig()
    tensorrt_bazel_config.add_config("TENSORRT_INCLUDE_PATH", tensorrt_include_path)
    tensorrt_bazel_config.add_config("TENSORRT_LIB_PATH", tensorrt_lib_path)
    tensorrt_bazel_config.save(config_file_path)
    print(f"Successfully saved TensorRT configuration to {config_file_path}")

if __name__ == "__main__":
    main()
