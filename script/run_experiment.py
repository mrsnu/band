#!/usr/bin/env python
from argparse import Namespace
import json
import os
import shutil

from utils import clean_bazel, get_argument_parser, get_platform, run_cmd
from run_benchmark import BASE_DIR, benchmark_android


def run(args):
    if args.experiment_name == 'balance':
        run_balance_effect(args)
    else:
        raise ValueError(args.experiment_name)


def run_balance_effect(args: Namespace):
    if args.rebuild:
        clean_bazel(args.docker)
    if os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    # Need to set Android build option in ./configure
    label_configs = [
        ('unbalanced', 'script/config_samples/benchmark_heat_unbalanced.json'),
        ('balanced', 'script/config_samples/benchmark_heat_balanced.json'),
        ]
    for label, config_path in label_configs:
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                        args.docker, config_path, run_as_su=args.run_as_su)
        with open(config_path) as fp:
            config_dict = json.load(fp)
            monitor_log_path = config_dict['monitor_log_path']
            log_path = config_dict['log_path']
        run_cmd(f'adb pull {monitor_log_path} {BASE_DIR}/monitor_log_{label}.json')
        run_cmd(f'adb pull {log_path} {BASE_DIR}/log_{label}.json')


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
