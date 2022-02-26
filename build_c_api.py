# base code borrowed from below
# https://github.com/asus4/tf-lite-unity-sample/blob/master/build_tflite.py

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import multiprocessing

BASE_DIR = 'bin'


def run_cmd(cmd):
    print(cmd)
    args = shlex.split(cmd)
    subprocess.call(args, cwd=os.getcwd())


def copy(src, dst):
    subprocess.call(['mkdir', '-p', f'{os.path.normpath(dst)}'])
    # append filename to dst directory
    dst = os.path.join(dst, os.path.basename(src))
    shutil.copyfile(src, dst)


def patch(file_path, target_str, patched_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()
    source = source.replace(target_str, patched_str)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(source)


def get_options(enable_xnnpack, debug):
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    debug_option = 'dbg' if debug else 'opt'
    return f'--jobs={multiprocessing.cpu_count()} -c {debug_option} --define tflite_with_xnnpack={option_xnnpack}'


def get_dst_path(platform, debug):
    build = 'debug' if debug else 'release'
    return os.path.join(BASE_DIR, platform, build)


def build_mac(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build --config=macos {get_options(enable_xnnpack, debug)} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib',
         get_dst_path('darwin', debug))

    # Metal Delegate
    # v2.3.0 or later, Need to apply the following patch to build metal delegate
    # For further info
    # https://github.com/tensorflow/tensorflow/issues/41039#issuecomment-664701908
    cpuinfo_file = f'{TENSORFLOW_PATH}/third_party/cpuinfo/BUILD.bazel'
    original = '"cpu": "darwin",'
    patched = '"cpu": "darwin_x86_64",'
    patch(cpuinfo_file, original, patched)
    # Build Metal Delegate
    run_cmd('bazel build --config=macos -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=default --linkopt -s --strip always --apple_platform_type=macos //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib')
    copy('bazel-bin/tensorflow/lite/delegates/gpu/tensorflow_lite_gpu_dylib.dylib',
         get_dst_path('darwin', debug))
    # Restore it
    patch(cpuinfo_file, patched, original)


def build_windows(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build {get_options(enable_xnnpack, debug)} tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/tensorflowlite_c.dll',
         get_dst_path('windows', debug))
    if debug:
        copy('bazel-bin/tensorflow/lite/c/tensorflowlite_c.pdb',
             get_dst_path('windows', debug))


def build_linux(debug=False):
    run_cmd(
        f'bazel build {get_options(False, debug)} --cxxopt=--std=c++11 tensorflow/lite/c:tensorflowlite_c')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so',
         get_dst_path('linux', debug))


def build_ios(debug=False):
    run_cmd(
        f'bazel build --config=ios_fat {get_options(False, debug)} //tensorflow/lite/ios:TensorFlowLiteC_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteC_framework.zip', 'iOS')
    # Metal Delegate
    run_cmd(
        f'bazel build {get_options(False, debug)} --config=ios_fat //tensorflow/lite/ios:TensorFlowLiteCMetal_framework')
    unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCMetal_framework.zip', 'iOS')
    # CoreML Delegate
    # run_cmd('bazel build -c opt --config=ios_fat //tensorflow/lite/ios:TensorFlowLiteCCoreML_framework')
    # unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteCCoreML_framework.zip', 'iOS')
    # SelectOps Delegate
    # run_cmd('bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework')
    # unzip('bazel-bin/tensorflow/lite/ios/TensorFlowLiteSelectTfOps_framework.zip', 'iOS')


def build_android(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build --config=android_arm64 {get_options(False, debug)} //tensorflow/lite/c:libtensorflowlite_c.so')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so',
         get_dst_path('armv8-a', debug))
    run_cmd(
        f'bazel build --config=android_arm {get_options(False, debug)} //tensorflow/lite/c:libtensorflowlite_c.so')
    copy('bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so',
         get_dst_path('armeabi-v7a', debug))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update TensorFlow Lite libraries for Unity')
    parser.add_argument('-macos', action="store_true", default=False,
                        help='Build macOS')
    parser.add_argument('-windows', action="store_true", default=False,
                        help='Build Windows')
    parser.add_argument('-linux', action="store_true", default=False,
                        help='Build Linux')
    parser.add_argument('-ios', action="store_true", default=False,
                        help='Build iOS')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Build Android')
    parser.add_argument('-xnnpack', action="store_true", default=False,
                        help='Build with XNNPACK')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Build debug (default = release)')

    args = parser.parse_args()

    platform_name = platform.system()

    if args.macos:
        assert platform_name == 'Darwin', f'-macos not suppoted on the platfrom: {platform_name}'
        print('Build macOS')
        build_mac(args.xnnpack, args.debug)

    if args.windows:
        assert platform_name == 'Windows', f'-windows not suppoted on the platfrom: {platform_name}'
        print('Build Windows')
        build_windows(args.xnnpack, args.debug)

    if args.linux:
        assert platform_name == 'Linux', f'-linux not suppoted on the platfrom: {platform_name}'
        print('Build Linux')
        build_linux(args.debug)

    if args.ios:
        assert platform_name == 'Darwin', f'-ios not suppoted on the platfrom: {platform_name}'
        # Need to set iOS build option in ./configure
        print('Build iOS')
        build_ios(args.debug)

    if args.android:
        # Need to set Android build option in ./configure
        print('Build Android')
        build_android(args.xnnpack, args.debug)
