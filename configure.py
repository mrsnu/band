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


def norm_path(path):
    if path[-1] == "/":
        return path[:-1]
    else:
        return path


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


def validate_cuda(cuda_home_path):
    if not os.path.exists(cuda_home_path):
        return False
    if not os.path.exists(os.path.join(cuda_home_path, "include")):
        return False
    if not os.path.exists(os.path.join(cuda_home_path, "lib64")):
        return False
    if not os.path.exists(os.path.join(cuda_home_path, "lib64", "libcudart.so")):
        return False
    if not os.path.exists(os.path.join(cuda_home_path, "lib64", "libcublas.so")):
        return False
    return True


def validate_cudnn(cudnn_home_path):
    if not os.path.exists(cudnn_home_path):
        return False
    if not os.path.exists(os.path.join(cudnn_home_path, "include")):
        return False
    if not os.path.exists(os.path.join(cudnn_home_path, "lib64")):
        return False
    if not os.path.exists(os.path.join(cudnn_home_path, "lib64", "libcudnn.so")):
        return False
    
    
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
        
    def get_config(self, env_var):
        return self.build_configs.get(env_var, None)

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
    
    # CUDA include & lib path
    cuda_version = get_var("CUDA_VERSION", "11.2")
    cuda_home_path = get_var("CUDA_HOME", "/usr/local/cuda")
    config_file_path = os.path.join(ROOT_DIR, ".band_cuda_config.bazelrc")
    cuda_bazel_config = BazelConfig()
    cuda_bazel_config.add_config("CUDA_HOME", cuda_home_path)
    cuda_bazel_config.add_config("CUDA_BIN_PATH", os.path.join(cuda_home_path, "bin"))
    
    # Include paths
    cuda_bazel_config.add_config("CUDA_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUBLAS_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUPTI_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUSOLVER_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUSPARSE_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUFFT_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CURAND_INCLUDE_PATH", os.path.join(cuda_home_path, "include"))
    cuda_bazel_config.add_config("CUDNN_INCLUDE_PATH", get_var("CUDNN_INCLUDE_PATH", "/usr/include"))
    cuda_bazel_config.add_config("TENSORRT_INCLUDE_PATH", get_var("TENSORRT_INCLUDE_PATH", "/usr/include/x86_64-linux-gnu"))
    
    # Lib paths
    cuda_bazel_config.add_config("CUDA_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CUBLAS_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CUPTI_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CUSOLVER_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CUSPARSE_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CUFFT_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("CURAND_LIB_PATH", os.path.join(cuda_home_path, "lib64"))
    cuda_bazel_config.add_config("NVVM_LIB_PATH", os.path.join(cuda_home_path, "nvvm/libdevice"))
    cuda_bazel_config.add_config("CUDNN_LIB_PATH", get_var("CUDNN_LIB_PATH", "/usr/lib/x86_64-linux-gnu"))
    cuda_bazel_config.add_config("TENSORRT_LIB_PATH", get_var("TENSORRT_LIB_PATH", "/usr/lib/x86_64-linux-gnu"))
    
    cuda_bazel_config.save(config_file_path)
    print(f"Successfully saved CUDA configuration to {config_file_path}")

if __name__ == "__main__":
    main()
