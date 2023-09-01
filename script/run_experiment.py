#!/usr/bin/env python
from argparse import Namespace
import json
import os
import shutil

from utils import clean_bazel, get_argument_parser, get_platform, run_cmd, run_on_android
from utils.dvfs import fix_all_cpufreq_max, set_all_cpu_governor_schedutil
from run_benchmark import benchmark_android, LOCAL_BENCHMARK_DIRNAME


def run(args):
    if args.experiment_name == 'balance':
        run_balance_effect(args)
    else:
        raise ValueError(args.experiment_name)


def run_balance_effect(args: Namespace):
    fix_all_cpufreq_max()
    if args.rebuild:
        clean_bazel(args.docker)
    if os.path.isdir(LOCAL_BENCHMARK_DIRNAME):
        shutil.rmtree(LOCAL_BENCHMARK_DIRNAME)
    # Need to set Android build option in ./configure
    label_configs = [
        ('unbalanced', 'script/config_samples/benchmark_heat_unbalanced.json'),
        ('balanced', 'script/config_samples/benchmark_heat_balanced.json'),
        ]
    for label, config_path in label_configs:
        with open(config_path) as fp:
            config_dict = json.load(fp)
            monitor_log_path = config_dict['monitor_log_path']
            log_path = config_dict['log_path']
        run_on_android(f'rm -rf {monitor_log_path} {log_path}',
                       run_as_su=args.run_as_su)
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                        args.docker, config_path, run_as_su=args.run_as_su)
        run_cmd(f'adb pull {monitor_log_path} {LOCAL_BENCHMARK_DIRNAME}/monitor_log_{label}.json')
        run_cmd(f'adb pull {log_path} {LOCAL_BENCHMARK_DIRNAME}/log_{label}.json')
    set_all_cpu_governor_schedutil()


def get_args():
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.set_defaults(
        docker=True,
        android=True,
        )
    parser.add_argument('experiment_name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
