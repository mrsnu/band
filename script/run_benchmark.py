#!/usr/bin/env python
import argparse
import tempfile
import shutil

from utils import *

BASE_DIR = 'benchmark'
TARGET = "band/tool:band_benchmark"

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
    elif platform.system() == 'Linux':
        run_cmd(build_command)
        copy('bazel-bin/band/tool/band_benchmark', target_base_dir)

    shutil.copy(config_path, f'{target_base_dir}/config.json')
    push_to_android(f'{target_base_dir}', '')
    run_binary_android('', f'benchmark/band_benchmark',
                       'benchmark/config.json')


if __name__ == '__main__':
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.add_argument('-c', '--config', default='band/test/data/benchmark_config.json',
                        help='Target config (default = band/test/data/benchmark_config.json)')
    args = parser.parse_args()

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
        benchmark_local(args.debug, args.trace, get_platform(), args.backend, args.build, args.config)
