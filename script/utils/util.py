from genericpath import isfile
import os
import argparse
import platform as plf
import shlex
import shutil
import subprocess
import multiprocessing
from .docker import run_cmd_docker

PLATFORM = {
    "linux": "linux_x86_64",
    "windows": "windows",
    "android": "android_arm64",
}


def is_windows():
    return plf.system() == "Windows"


def is_linux():
    return plf.system() == "Linux"


def is_cygwin():
    return plf.system().startswith("CYGWIN_NT")


def get_platform():
    return plf.system().lower()


def canon_path(path):
    if is_windows() or is_cygwin():
        return path.replace('\\', '/')
    return path


def run_cmd(cmd):
    args = shlex.split(cmd)
    subprocess.call(args, cwd=os.getcwd())


def copy(src, dst):
    subprocess.call(['mkdir', '-p', f'{os.path.normpath(dst)}'])
    # append filename to dst directory

    dst = canon_path(os.path.join(dst, os.path.basename(src)))

    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        try:
            shutil.copy(src, dst)
        except FileNotFoundError:
            print(dst)
            shutil.copy(src + ".exe", dst + ".exe")


def get_bazel_options(
        debug: bool,
        trace: bool, 
        platform: str,
        backend: str):
    opt = ""
    opt += "-c " + ("dbg" if debug else "opt") + " "
    opt += "--strip " + ("never" if debug else "always") + " "
    opt += f"--config {PLATFORM[platform]}" + ("" if backend == "none" else f"_{backend}") + ("" if trace == False else " --config trace") + " "
    return opt


def make_cmd(
        build_only: bool, 
        debug: bool,
        trace: bool, 
        platform: str, 
        backend: str, 
        target: str,
    ):
    cmd = "bazel" + " "
    cmd += ("build" if build_only else "test") + " "
    cmd += get_bazel_options(debug, trace, platform, backend)
    cmd += target
    return cmd


def get_dst_path(base_dir, target_platform, debug):
    build = 'debug' if debug else 'release'
    path = os.path.join(base_dir, target_platform, build)
    return canon_path(path)


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
    parser.add_argument('-B', '--backend', type=str,
                        default='all', help='Backend <all|none>')
    parser.add_argument('-android', action="store_true", default=False,
                        help='Test on Android (with adb)')
    parser.add_argument('-docker', action="store_true", default=False,
                        help='Compile / Pull cross-compiled binaries for android from docker (assuming that the current docker engine has devcontainer built with a /.devcontainer')
    parser.add_argument('-s', '--ssh', required=False, default=None, help="SSH host name (e.g. dev@ssh.band.org)")
    parser.add_argument('-t', '--trace', action="store_true", default=True, help='Build with trace (default = True)')
    parser.add_argument('-d', '--debug', action="store_true", default=False,
                        help='Build debug (default = release)')
    parser.add_argument('-r', '--rebuild', action="store_true", default=False,
                        help='Re-build test target, only affects to android ')
    parser.add_argument('-b', '--build', action="store_true", default=False,
                        help='Build only, only affects to local (default = false)')
    return parser


def clean_bazel(docker):
    if docker:
        run_cmd_docker('bazel clean')
    else:
        run_cmd('bazel clean')
