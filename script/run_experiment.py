#!/usr/bin/env python
from argparse import Namespace
import json
import os
import shutil
import subprocess
from time import sleep

from utils import clean_bazel, get_argument_parser, get_platform, run_cmd, run_on_android
import utils.dvfs as dvfs
from run_benchmark import benchmark_android, LOCAL_BENCHMARK_DIRNAME


def shutdown_thermal_engine():
    run_on_android('stop vendor.thermal-engine', run_as_su=args.run_as_su)


def start_thermal_engine():
    run_on_android('start vendor.thermal-engine', run_as_su=args.run_as_su)


def run(args):
    if args.experiment_name == 'balance':
        run_balance_effect(args)
    else:
        raise ValueError(args.experiment_name)


def get_reference_temperature(args):
    out = run_on_android('cat /sys/class/thermal/tz-by-name/sdm-therm/temp',
                   capture_output=True, run_as_su=args.run_as_su)
    return float(out) / 1000


def busywait(seconds):
    shcmd = (
        'ts=`date +%s`; tn=`date +%N`; '
        f'while [ `date +%s%N` -lt $(($ts + {seconds}))$tn ]; do :; done'
        )
    subprocess.check_call(f'adb shell "{shcmd}"')


def wait_until_temperature(args):
    ref_temp = get_reference_temperature(args)
    is_cooling = ref_temp > args.init_temperature
    print('ref_temp:', ref_temp, 'is_cooling:', is_cooling)
    if is_cooling:
        dvfs.fix_all_cpufreq_min()
    else:
        dvfs.fix_all_cpufreq_max()
    while True:
        ref_temp = get_reference_temperature(args)
        print('ref_temp:', ref_temp, end='\r')
        if is_cooling:
            if ref_temp > args.init_temperature:
                sleep(1)
            else:
                break
        else:
            if ref_temp < args.init_temperature:
                busywait(seconds=1)
            else:
                break
    print()
    dvfs.set_all_cpu_governor_schedutil()


def run_balance_effect(args: Namespace):
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
        wait_until_temperature(args)
        shutdown_thermal_engine()
        dvfs.fix_all_cpufreq_max()
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                        args.docker, config_path, run_as_su=args.run_as_su)
        dvfs.set_all_cpu_governor_schedutil()
        start_thermal_engine()
        run_cmd(f'adb pull {monitor_log_path} {LOCAL_BENCHMARK_DIRNAME}/monitor_log_{label}.json')
        run_cmd(f'adb pull {log_path} {LOCAL_BENCHMARK_DIRNAME}/log_{label}.json')


def get_args():
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.add_argument('--init-temperature', type=float, default=35)
    parser.set_defaults(docker=True, android=True)
    parser.add_argument('experiment_name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
