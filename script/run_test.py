#!/usr/bin/env python
import argparse
import platform as plf
import tempfile

from utils import *

BASE_DIR = 'test_bin'
TARGET = "//band/test/..."

def test_android(debug, platform, backend, docker, filter=""):
    # build android targets only (specified in band_cc_android_test tags)
    build_command = make_cmd(
        build_only=True,
        debug=debug,
        platform=platform,
        backend=backend,
        target=TARGET,
    )
    
    if docker:
        run_cmd_docker(build_command)
        # Create a local path
        subprocess.call(
            ['mkdir', '-p', f'{get_dst_path(BASE_DIR, "armv8-a", debug)}'])
        copy_docker(f'bazel-bin/band/test {get_dst_path(BASE_DIR, "armv8-a", debug)}')
    elif plf.system() == "Linux":
        run_cmd(f'{build_command}')
        copy('bazel-bin/band/test', get_dst_path(BASE_DIR, "armv8-a", debug))
    else:
        raise ValueError(f"Unable to build from the given platform {plf.system()}. Maybe you want to build with docker or remote?")

    temp_dir_name = next(tempfile._get_candidate_names())
    print("Copy test data to device")
    push_to_android('band/test', f'{temp_dir_name}/band/test')
    for test_file in os.listdir(f'{get_dst_path(BASE_DIR, "armv8-a", debug)}/test'):
        if filter != "":
            if filter not in test_file:
                continue
        # Check whether the given path is binary and file
        if test_file.find('.') == -1 and os.path.isfile(f'{get_dst_path(BASE_DIR, "armv8-a", debug)}/test/{test_file}'):
            print(f'-----------TEST : {test_file}-----------')
            push_to_android(
                f'{get_dst_path(BASE_DIR, "armv8-a", debug)}/test/{test_file}', temp_dir_name)
            run_binary_android(f'{temp_dir_name}/', f'{test_file}')

    print("Clean up test directory from device")
    run_cmd(
        f'adb -d shell rm -r /data/local/tmp/{temp_dir_name}')


if __name__ == '__main__':
    parser = get_argument_parser('Run Band tests for specific target platform')
    parser.add_argument('-f', '--filter', default="",
                        help='Run specific test that contains given string (only for android)')

    args = parser.parse_args()
    
    print(f"Test {args.platform}")
    if args.rebuild: 
        run_cmd("bazel clean")
    
    if args.android:
        run_cmd()
        test_android(args.debug, args.platform, args.backend, args.docker, args.filter)
    else:
        run_cmd(
            make_cmd(
                args.build, 
                args.debug, 
                args.platform,
                args.backend, 
                TARGET
            )
        )
