from genericpath import isfile
import os
import argparse
import platform
import shlex
import shutil
import subprocess
import multiprocessing

PLATFORM = {
    "linux": "linux_x86_64",
    "windows": "windows",
    "android": "android_arm64",
}

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


def get_bazel_options(
        debug: bool,
        platform: str,
        backend: str):
    opt = ""
    opt += "-c " + ("dbg" if debug else "opt") + " "
    opt += "--strip " + ("never" if debug else "always") + " "
    opt += "--config " + f"{PLATFORM[platform]}" + ("" if backend == "none" else f"_{backend}") + " "
    return opt


def make_cmd(
        build_only: bool, 
        debug: bool, 
        platform: str, 
        backend: str, 
        target: str,
    ):
    cmd = "bazel" + " "
    cmd += "build" if build_only else "test" + " "
    cmd += get_bazel_options(debug, platform, backend)
    cmd += target
    return cmd
    

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

def get_argument_parser(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--platform', type=str, required=True, help='Platform <linux|android|windows>')
    parser.add_argument('-B', '--backend', type=str, required=True, help='Backend <tflite|none>')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Test on Android (with adb)')
    # TODO: add support for arbitrary docker container name and directory
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries for android from docker (assuming that the current docker context has devcontainer built with a /.devcontainer')
    parser.add_argument('-s', '--ssh', required=False, default=None, help="SSH host name (e.g. dev@ssh.band.org)")
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android ')
    parser.add_argument('-b', '--build', action="store_true", default=False,
                        help='Build only, only affects to local (default = false)')
    return parser