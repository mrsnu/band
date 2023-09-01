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

import shutil

from utils import *

ANDROID_BENCHMARK_DIRPATH = f'{ANDROID_BASE}benchmark'
LOCAL_BENCHMARK_DIRNAME = 'benchmark'
TARGET = "band/tool:band_benchmark"
DEFAULT_CONFIG = 'script/config_samples/benchmark_config.json'


def benchmark_local(debug, trace, platform, backend, build_only, config_path):
    build_cmd = make_cmd(
        build_only=build_only,
        debug=debug,
        trace=trace,
        platform=platform,
        backend=backend,
        target=TARGET
    )
    run_cmd(build_cmd)
    copy('bazel-bin/band/tool/band_benchmark', LOCAL_BENCHMARK_DIRNAME)
    run_cmd(f'chmod 777 {LOCAL_BENCHMARK_DIRNAME}/band_benchmark')
    if platform == "linux":
        run_cmd(f'{LOCAL_BENCHMARK_DIRNAME}/band_benchmark {config_path}')
    else:
        run_cmd(f'{LOCAL_BENCHMARK_DIRNAME}/band_benchmark.exe {config_path}')


def benchmark_android(debug, trace, platform, backend, docker, config_path="",
                      run_as_su=False):
    # build android targets only (specified in band_cc_android_test tags)

    build_command = make_cmd(
        build_only=True,
        debug=debug,
        trace=trace,
        platform="android",
        backend=backend,
        target=TARGET,
    )
    if docker:
        run_cmd_docker(build_command)
        # create a local path
        subprocess.check_call(['mkdir', '-p', f'{LOCAL_BENCHMARK_DIRNAME}'])
        # run_cmd(
        #     f'sh script/docker_util.sh -d bazel-bin/band/tool/band_benchmark {LOCAL_BENCHMARK_DIRNAME}')
        copy_docker('bazel-bin/band/tool/band_benchmark', LOCAL_BENCHMARK_DIRNAME)
    elif platform.system() == 'Linux':
        run_cmd(build_command)
        copy('bazel-bin/band/tool/band_benchmark', LOCAL_BENCHMARK_DIRNAME)

    config_paths = []
    if os.path.isfile(config_path):
        config_paths.append(config_path)
    elif os.path.isdir(config_path):
        for file in os.listdir(config_path):
            if file.endswith('.json'):
                config_paths.append(f'{config_path}/{file}')

    for config_path in config_paths:
        name = os.path.basename(config_path)
        shutil.copy(config_path, f'{LOCAL_BENCHMARK_DIRNAME}/{name}')
        print (f'Push {name} to Android')

    run_on_android(f'rm -rf {ANDROID_BASE}{LOCAL_BENCHMARK_DIRNAME}',
                   run_as_su=run_as_su)
    push_to_android(f'{LOCAL_BENCHMARK_DIRNAME}', '')

    for config_path in config_paths:
        name = os.path.basename(config_path)
        print(f'Run {name}')
        android_benchmark_dirnatm = ANDROID_BENCHMARK_DIRPATH.split('/')[-1]
        run_binary_android(
            android_benchmark_dirnatm, 'band_benchmark', name,
            run_as_su=run_as_su)


if __name__ == '__main__':
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.add_argument('-c', '--config', default=f'{DEFAULT_CONFIG}',
                        help=f'Target config file or directory (default = {DEFAULT_CONFIG})')
    args = parser.parse_args()

    # copy from band/test/data/benchmark_config.json 
    if args.config == DEFAULT_CONFIG:
        if args.android:
            shutil.copy('script/config_samples/benchmark_heft.json', f'{DEFAULT_CONFIG}')
        else:
            args.config = 'band/test/data/benchmark_config.json'


    if args.rebuild:
        clean_bazel(args.docker)

    if os.path.isdir(ANDROID_BENCHMARK_DIRPATH):
        shutil.rmtree(ANDROID_BENCHMARK_DIRPATH)

    if args.android:
        # Need to set Android build option in ./configure
        print('Benchmark Android')
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                          args.docker, args.config, run_as_su=True)
    else:
        print(f'Benchmark {get_platform()}')
        benchmark_local(args.debug, args.trace, get_platform(),
                        args.backend, args.build, args.config)
