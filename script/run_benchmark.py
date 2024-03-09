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
import tempfile
import shutil

from utils import *

BASE_DIR = 'benchmark'
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
    copy('bazel-bin/band/tool/band_benchmark', BASE_DIR)
    run_cmd(
        f'chmod 777 {BASE_DIR}/band_benchmark'
    )
    if platform == "linux":
        run_cmd(
            f'{BASE_DIR}/band_benchmark {config_path}'
        )
    else:
        run_cmd(
            f'{BASE_DIR}/band_benchmark.exe {config_path}'
        )


def benchmark_android(debug, trace, platform, backend, docker, config_path=""):
    target_base_dir = BASE_DIR
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
        subprocess.call(['mkdir', '-p', f'{target_base_dir}'])
        # run_cmd(
        #     f'sh script/docker_util.sh -d bazel-bin/band/tool/band_benchmark {target_base_dir}')
        copy_docker('bazel-bin/band/tool/band_benchmark', target_base_dir)
    elif platform == 'linux':
        run_cmd(build_command)
        copy('bazel-bin/band/tool/band_benchmark', target_base_dir)

    config_paths = []
    if os.path.isfile(config_path):
        config_paths.append(config_path)
    elif os.path.isdir(config_path):
        for file in os.listdir(config_path):
            if file.endswith('.json'):
                config_paths.append(f'{config_path}/{file}')

    for config_path in config_paths:
        name = os.path.basename(config_path)
        os.makedirs(f'{target_base_dir}', exist_ok=True)
        shutil.copy(config_path, f'{target_base_dir}/{name}')
        print (f'Push {name} to Android')

    push_to_android(f'{target_base_dir}', '')

    for config_path in config_paths:
        name = os.path.basename(config_path)
        print(f'Run {name}')
        run_binary_android('', f'{target_base_dir}/band_benchmark',
                           f'{target_base_dir}/{name}')


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

    if os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    if args.android:
        # Need to set Android build option in ./configure
        print('Benchmark Android')
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                          args.docker, args.config)
    else:
        print(f'Benchmark {get_platform()}')
        benchmark_local(args.debug, args.trace, get_platform(),
                        args.backend, args.build, args.config)