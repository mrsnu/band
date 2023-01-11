# base code borrowed from below
# https://github.com/asus4/tf-lite-unity-sample/blob/master/build_tflite.py

import argparse
import os
import platform
import shutil
import shlex
import subprocess
import multiprocessing

BASE_DIR = 'bin'

def run_cmd(cmd):
    args = shlex.split(cmd)
    subprocess.call(args, cwd=os.getcwd())

def copy(src, dst):
    subprocess.call(['mkdir', '-p', f'{os.path.normpath(dst)}'])
    # append filename to dst directory
    dst = os.path.join(dst, os.path.basename(src))
    shutil.copytree(src, dst)

def get_options(enable_xnnpack, debug):
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    debug_option = 'dbg' if debug else 'opt'
    strip_option = 'never' if debug else 'always'
    return f'--jobs={multiprocessing.cpu_count()} {" --test_output=all" if debug else ""} -c {debug_option} --strip {strip_option} --config tflite --define tflite_with_xnnpack={option_xnnpack}'

def get_dst_path(target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(BASE_DIR, target_platform, build)
    if platform.system() == 'Windows':
        path = path.replace('\\', '/')
    return path

def build_windows(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build {get_options(enable_xnnpack, debug)} band/c:band_c')
    copy('bazel-bin/band/c/band_c.dll',
         get_dst_path('windows', debug))
    if debug:
        copy('bazel-bin/band/c/band_c.pdb',
             get_dst_path('windows', debug))


def build_linux(debug=False):
    run_cmd(
        f'bazel build {get_options(False, debug)} --cxxopt=--std=c++11 band/c:band_c')
    copy('bazel-bin/band/c/libband_c.so',
         get_dst_path('linux', debug))


def build_android(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build --config=android_arm64 --strip always {get_options(enable_xnnpack, debug)} band/c:band_c')
    copy('bazel-bin/band/c/libband_c.so',
         get_dst_path('armv8-a', debug))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build Band c apis for specific target platform')
    parser.add_argument('-windows', action="store_true", default=False,
                        help='Build Windows')
    parser.add_argument('-linux', action="store_true", default=False,
                        help='Build Linux')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Build Android')
    parser.add_argument('-xnnpack', action="store_true", default=False,
                        help='Build with XNNPACK')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Build debug (default = release)')

    args = parser.parse_args()

    platform_name = platform.system()

    if args.windows:
        assert platform_name == 'Windows', f'-windows not suppoted on the platfrom: {platform_name}'
        print('Build Windows')
        build_windows(args.xnnpack, args.debug)

    if args.linux:
        assert platform_name == 'Linux', f'-linux not suppoted on the platfrom: {platform_name}'
        print('Build Linux')
        build_linux(args.debug)

    if args.android:
        # Need to set Android build option in ./configure
        print('Build Android')
        build_android(args.xnnpack, args.debug)
