#!/usr/bin/env python
# base code borrowed from below
# https://github.com/asus4/tf-lite-unity-sample/blob/master/build_tflite.py

import argparse
from util import *

BASE_DIR = 'bin'


def build_windows(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build {get_bazel_options(enable_xnnpack, debug)} band/c:band_c')
    copy('bazel-bin/band/c/band_c.dll',
         get_dst_path(BASE_DIR, 'windows', debug))
    if debug:
        copy('bazel-bin/band/c/band_c.pdb',
             get_dst_path(BASE_DIR, 'windows', debug))


def build_linux(debug=False):
    run_cmd(
        f'bazel build {get_bazel_options(False, debug)} --cxxopt=--std=c++11 band/c:band_c')
    copy('bazel-bin/band/c/libband_c.so',
         get_dst_path(BASE_DIR, 'linux', debug))


def build_android(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel build {get_bazel_options(enable_xnnpack, debug, True)} band/c:band_c')
    copy('bazel-bin/band/c/libband_c.so',
         get_dst_path(BASE_DIR, 'armv8-a', debug))


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
    parser.add_argument('-d', '--debug', action="store_true", default=False,
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
