from genericpath import isfile
import os
import platform
import shlex
import shutil
import subprocess
import multiprocessing


def run_cmd(cmd):
    args = shlex.split(cmd)
    subprocess.call(args, cwd=os.getcwd())


def copy(src, dst):
    subprocess.call(['mkdir', '-p', f'{os.path.normpath(dst)}'])
    # append filename to dst directory
    dst = os.path.join(dst, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def get_bazel_options(enable_xnnpack: bool, debug: bool, android: bool = False):
    option_xnnpack = 'true' if enable_xnnpack else 'false'
    build_option = 'dbg' if debug else 'opt'
    strip_option = 'never' if debug else 'always'
    android_option = '--config=android_arm64' if android else ''
    test_output_option = "--test_output=all" if debug else ""
    return f'{android_option} --jobs={multiprocessing.cpu_count()} {test_output_option} -c {build_option} --strip {strip_option} --config tflite --define tflite_with_xnnpack={option_xnnpack}'


def get_dst_path(base_dir, target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(base_dir, target_platform, build)
    if platform.system() == 'Windows':
        path = path.replace('\\', '/')
    return path


ANDROID_BASE = '/data/local/tmp/'


def push_to_android(src, dst):
    run_cmd(f'adb -d push {src} {ANDROID_BASE}{dst}')


def run_binary_android(basepath, path, option=''):
    run_cmd(f'adb -d shell chmod 777 {ANDROID_BASE}{basepath}{path}')
    # cd & run -- to preserve the relative relation of the binary
    run_cmd(
        f'adb -d shell cd {ANDROID_BASE}{basepath} && ./{path} {option}')
