#!/usr/bin/env python
# base code borrowed from below
# https://github.com/asus4/tf-lite-unity-sample/blob/master/build_tflite.py

import argparse
from utils import *

BASE_DIR = 'bin'
TARGET = 'band/c:band_c_pkg'

if __name__ == '__main__':
    parser = get_argument_parser(
        desc='Build Band C apis for specific target platform')
    args = parser.parse_args()
    platform = "android" if args.android else get_platform()

    build_cmd = make_cmd(
        build_only=True,
        debug=args.debug,
        trace=args.trace,
        platform="android" if args.android else get_platform(),
        backend=args.backend,
        target=TARGET
    )
    if args.docker:
        run_cmd_docker(build_cmd)
    else:
        run_cmd(build_cmd)
    
    copy('bazel-bin/band/c/band_c_pkg.tar',
            get_dst_path(BASE_DIR, platform, args.debug))
