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
