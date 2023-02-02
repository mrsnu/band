
import argparse
import tempfile

from util import *

BASE_DIR = 'test_bin'


def test_local(enable_xnnpack=False, debug=False, build_only=False):
    target_option = 'build' if build_only else 'test'
    run_cmd(
        f'bazel {target_option} {get_bazel_options(enable_xnnpack, debug)} band/test/...')


def test_android(enable_xnnpack=False, debug=False, docker=False, rebuild=False, filter=""):
    # build android targets only (specified in band_cc_android_test tags)
    build_command = f'{"bazel clean &&" if rebuild else ""} bazel build {get_bazel_options(enable_xnnpack, debug, True)} band/test/...'
    if docker:
        run_cmd(f'sh script/docker_util.sh -r {build_command}')
        # create a local path
        subprocess.call(
            ['mkdir', '-p', f'{get_dst_path(BASE_DIR, "armv8-a", debug)}'])
        run_cmd(
            f'sh script/docker_util.sh -d bazel-bin/band/test {get_dst_path(BASE_DIR, "armv8-a", debug)}')
    elif platform.system() == 'Linux':
        run_cmd(f'{build_command}')
        copy('bazel-bin/band/test', get_dst_path(BASE_DIR, "armv8-a", debug))

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
    platform_name = platform.system()
    parser = argparse.ArgumentParser(
        description=f'Run Band tests for specific target platform (default: {platform_name})')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Test on Android (with adb)')
    # TODO: add support for arbitrary docker container name and directory
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries for android from docker (assuming that the current docker context has devcontainer built with a /.devcontainer')
    # TODO: add support for arbitrary ssh endpoint
    parser.add_argument('-xnnpack', action="store_true", default=False,
                        help='Build with XNNPACK')
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-b', '--build', action="store_true", default=False,
                        help='Build only, only affects to local (default = false)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android ')
    parser.add_argument('-f', '--filter', default="",
                        help='Run specific test that contains given string (only for android)')

    args = parser.parse_args()

    if args.android:
        # Need to set Android build option in ./configure
        print('Test Android')
        test_android(args.xnnpack, args.debug, args.docker,
                     args.rebuild, args.filter)
    else:
        print(f'Test {platform_name}')
        test_local(args.xnnpack, args.debug, args.build)
