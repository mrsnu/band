#!/usr/bin/env python
import json
import shutil

import matplotlib.pyplot as plt

from utils import *
from run_benchmark import BASE_DIR, benchmark_android


def plot_data(monitor_log_path):
    with open(monitor_log_path) as fp:
        data_dict = json.load(fp)
    plt.plot()


def run(args):
    if args.rebuild:
        clean_bazel(args.docker)

    if os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    # Need to set Android build option in ./configure
    benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                      args.docker, args.config, run_as_su=True)

    with open(args.config) as fp:
        config_dict = json.load(fp)
        monitor_log_path = config_dict['monitor_log_path']
    run_cmd(f'adb pull {monitor_log_path} {BASE_DIR}')


def get_args():
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.set_defaults(
        config='script/config_samples/benchmark_power_baseline.json',
        docker=True,
        android=True,
        )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
