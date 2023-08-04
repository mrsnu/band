#!/usr/bin/env python
import shutil

from utils import *
from run_benchmark import BASE_DIR, benchmark_android


if __name__ == '__main__':
    parser = get_argument_parser("Run Band benchmarks for specific target platform. "
                                 "The user should prepare/specify dependent files either absolute path or relative path depending on an execution mode. "
                                 "This script executes the benchmark from (Android: /data/local/tmp, Other: current working directory)")
    parser.set_defaults(
        config='script/config_samples/benchmark_power_baseline.json',
        docker=True,
        android=True,
        )
    args = parser.parse_args()

    if args.rebuild:
        clean_bazel(args.docker)

    if os.path.isdir(BASE_DIR):
        shutil.rmtree(BASE_DIR)

    # Need to set Android build option in ./configure
    benchmark_android(args.debug, args.trace, get_platform(), args.backend,
                      args.docker, args.config, run_as_su=True)
