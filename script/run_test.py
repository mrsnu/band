
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
    return f'--jobs={multiprocessing.cpu_count()} -c {debug_option} --config tflite --define tflite_with_xnnpack={option_xnnpack}'

def get_dst_path(platform, debug):
    build = 'debug' if debug else 'release'
    return os.path.join(BASE_DIR, platform, build)

def test_local(enable_xnnpack=False, debug=False):
    run_cmd(
        f'bazel test {get_options(enable_xnnpack, debug)} band/test/...')

def test_android(enable_xnnpack=False, debug=False, docker=False):
    if docker:
        run_cmd(f'')
        run_cmd(f'script/download.sh bazel-bin/band/test bin/test')

    run_cmd(
        f'bazel build --config=android_arm64 --strip always  {get_options(False, debug)} band/test/...')

if __name__ == '__main__':
    platform_name = platform.system()
    parser = argparse.ArgumentParser(
        description=f'Run Band tests for specific target platform (default: {platform_name})')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Test on Android (with adb)')
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Pull cross-compiled binaries for android from docker (assuming that the current docker context has devcontainer built with a /.devcontainer')
    parser.add_argument('-xnnpack', action="store_true", default=False,
                        help='Build with XNNPACK')
    parser.add_argument('-debug', action="store_true", default=False,
                        help='Build debug (default = release)')

    args = parser.parse_args()

    if args.android:
        # Need to set Android build option in ./configure
        print('Test Android')
        test_android(args.xnnpack, args.debug, args.docker)
    else:
        print(f'Test {platform_name}')
        test_local(args.xnnpack, args.debug)
