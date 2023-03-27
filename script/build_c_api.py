#!/usr/bin/env python
# base code borrowed from below
# https://github.com/asus4/tf-lite-unity-sample/blob/master/build_tflite.py

import argparse
from utils import *

BASE_DIR = 'bin'
TARGET = 'band/c:band_c'


def copy_lib(debug, platform, android, docker):
    if platform == 'windows':
        copy('bazel-bin/band/c/band_c.dll',
             get_dst_path(BASE_DIR, 'windows', debug))
        if debug:
            copy('bazel-bin/band/c/band_c.pdb',
                 get_dst_path(BASE_DIR, 'windows', debug))
        return

    if android:
        if docker:
            copy_docker('bazel-bin/band/c/libband_c.so',
                        get_dst_path(BASE_DIR, 'armv8-a', debug))
        else:
            copy('bazel-bin/band/c/libband_c.so',
                 get_dst_path(BASE_DIR, 'armv8-a', debug))
        return

    if platform == "linux":
        copy('bazel-bin/band/c/libband_c.so',
             get_dst_path(BASE_DIR, 'linux', debug))
        return

    raise ValueError(f"Platform {platform} is not supported.")


if __name__ == '__main__':
    parser = get_argument_parser(
        desc='Build Band C apis for specific target platform')
    args = parser.parse_args()
    platform = "android" if args.android else get_platform()

    build_cmd = make_cmd(
        build_only=True,
        debug=args.debug,
        platform=platform,
        backend=args.backend,
        target=TARGET
    )
    if args.docker:
        run_cmd_docker(build_cmd)
    else:
        run_cmd(build_cmd)
    copy_lib(args.debug, platform, args.android, args.docker)
