#!/usr/bin/env python
from argparse import Namespace
import json
import os
import shutil
import subprocess
from time import sleep, time

from utils import clean_bazel, get_argument_parser, get_platform, run_cmd, run_on_android
from utils.dvfs import DVFS
from run_benchmark import benchmark_android, LOCAL_BENCHMARK_DIRNAME


dvfs = DVFS()


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


def stabalize_init_temperature(args):
    dvfs.set_cpufreq_all_governors('userspace')
    start_time = time()
    stabalization_start_time = -1
    ref_temp = get_reference_temperature(args)
    is_cooling = ref_temp > args.init_temperature
    while (stabalization_start_time < 0
           or time() - stabalization_start_time
            < args.temperature_stabalization_mins * 60):
        ref_temp = get_reference_temperature(args)
        if (stabalization_start_time < 0
            and ((is_cooling and ref_temp < args.init_temperature)
                 or (not is_cooling and ref_temp >= args.init_temperature))):
            stabalization_start_time = time()
        print(f'ref_temp: {ref_temp}, time: {int(time() - start_time)} s')
        if ref_temp > args.init_temperature:
            dvfs.fix_all_cpufreq_min()
            sleep(1)
        else:
            dvfs.fix_all_cpufreq_max()
            busywait(seconds=1)
    print()
    dvfs.set_cpufreq_all_governors('schedutil')


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
        stabalize_init_temperature(args)
        shutdown_thermal_engine()
        dvfs.set_cpufreq_all_governors('userspace')
        dvfs.fix_all_cpufreq_max()
        benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                        args.docker, config_path, run_as_su=args.run_as_su)
        dvfs.set_cpufreq_all_governors('schedutil')
        start_thermal_engine()
        run_cmd(f'adb pull {monitor_log_path} {LOCAL_BENCHMARK_DIRNAME}/monitor_log_{label}.json')
        run_cmd(f'adb pull {log_path} {LOCAL_BENCHMARK_DIRNAME}/log_{label}.json')


def get_args():
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.add_argument('--init-temperature', type=float, default=35)
    parser.add_argument('--temperature-stabalization-mins', type=float, default=10)
    parser.set_defaults(docker=True, android=True)
    parser.add_argument('experiment_name')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
