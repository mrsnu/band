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

import argparse
import platform as plf
import tempfile

from utils import *

BASE_DIR = 'test_bin'
TARGET = "band/test/..."

def get_target(debug, postfix=""):
    return f'{get_dst_path(BASE_DIR, "armv8-a", debug)}/' + postfix


if __name__ == '__main__':
    parser = get_argument_parser('Run Band tests for specific target platform')
    parser.add_argument('-f', '--filter', default="",
                        help='Run specific test that contains given string (only for android)')
    args = parser.parse_args()

    print(f"Test {get_platform()}")
    if args.rebuild: 
        clean_bazel(args.docker)
    
    if args.android:
        build_cmd = make_cmd(
                build_only=True,
                debug=args.debug,
                trace=args.trace,   
                platform='android',
                backend=args.backend,
                target=TARGET
            )
        subprocess.call(['mkdir', '-p', get_target(args.debug)])
        if args.docker:
            run_cmd_docker(build_cmd)
            copy_docker(f'bazel-bin/band/test', get_target(args.debug))
        else:
            run_cmd(build_cmd)
            copy(f'bazel-bin/band/test', get_target(args.debug))

        temp_dir_name = next(tempfile._get_candidate_names())
        print("Copy test data to device")
        push_to_android('band/test', f'{temp_dir_name}/band/test')
        for test_file in os.listdir(get_target(args.debug, 'test')):
            if args.filter != "":
                if args.filter not in test_file:
                    continue
            # Check whether the given path is binary and file
            if not '.' in test_file and os.path.isfile(get_target(args.debug, f'test/{test_file}')):
                print(f'-----------TEST : {test_file}-----------')
                push_to_android(
                    get_target(args.debug, f'test/{test_file}'), temp_dir_name)
                run_binary_android(f'{temp_dir_name}/', f'{test_file}')

        print("Clean up test directory from device")
        run_cmd(
            f'adb shell rm -r /data/local/tmp/{temp_dir_name}')
    else:
        cmd = make_cmd(
                args.build, 
                args.debug,
                args.trace,
                get_platform(),
                args.backend, 
                TARGET
            )
        run_cmd(cmd)
