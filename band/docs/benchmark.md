# Benchmark Tool for Band

Band provides a simple C++ binary to benchmark a runtime performance.
The binary generates repeatitive model requests based on a given config file, and reports latency statistics afterwards.

## How to run

`[root]/script/run_benchmark.py` script will build `band_benchmark` binary file and execute it with a specified config file. Built binary file and target config file can be found in `[root]/benchmark`.

### On Android
If you want to build binary from docker container (Refer to `[root]/script/docker_util.sh` for more detail)
```
python .\script\run_benchmark.py -android -docker -c .\benchmark_config.json
```
If you want to build locally
```
python .\script\run_benchmark.py -android -c .\benchmark_config.json
```


### On local desktop (Windows or Ubuntu)
```
python .\script\run_benchmark.py -c .\benchmark_config.json
```


## Config file

### Structure
* `models`: Models to run. For each model, specify the following fields.
  * `graph`: Model path.
  * `period_ms`: **Optional** The delay between subsequent requests in ms. The argument is only effective with `periodic` execution mode.
  * `batch_size`: The number of model requests in a frame. [default: 1]
  * `worker_id`: **Optional** Specify the worker id to run in int. The argument is only effective with `fixed_device` scheduler.
  * `slo_us` and `slo_scale`: **Optional** fields for specifying an SLO value for a model. Setting `slo_scale` will make the SLO = worst profiled latency of that model * `slo_scale`. `slo_scale` will be ignored if `slo_us` is given (i.e., no reason to specify both options).
* `log_path`: The log file path. (e.g., `/data/local/tmp/model_execution_log.csv`)
* `schedulers`: The scheduler types in `list[string]`. If N schedulers are specified, then N queues are generated.
  * `fixed_worker`
  * `round_robin`
  * `shortest_expected_latency`
  * `least_slack_time_first`
  * `heterogeneous_earliest_finish_time`
  * `heterogeneous_earliest_finish_time_reserved`
* `minimum_subgraph_size`: Minimum subgraph size. If candidate subgraph size is smaller than `minimum_subgraph_size`, the subgraph will not be created. [default: 7]
* `subgraph_preparation_type`: For schedulers using fallback, determine how to generate candidate subgraphs. [default: `merge_unit_subgraph`]
  * `no_fallback_subgraph`: Generate subgraphs per worker. Explicit fallback subgraph will not be generated.
  * `fallback_per_worker`: Generate fallback subgraphs for each worker.
  * `unit_subgraph`: Generate unit subgraphs considering all device supportiveness. All ops in same unit subgraph have same support devices.
  * `merge_unit_subgraph`: Add merged unit subgraphs to `unit_subgraph`.
* `execution_mode`: Specify a exeucution mode. Available execution modes are as follows:
  * `stream`: consecutively run batches.
  * `periodic`: invoke requests periodically.
  * `workload`: execute pre-defined sequence in `stream` manner based on a given workload file.
* `cpu_masks`: CPU cluster mask to set CPU affinity. [default: `ALL`]
  * `ALL`: All Cluster
  * `LITTLE`: LITTLE Cluster only
  * `BIG`: Big Cluster only
  * `PRIMARY`: Primary Core only
* `num_threads`: Number of computing threads for CPU delegates. [default: -1]
* `planner_cpu_masks`: CPU cluster mask to set CPU affinity of planner. [default: same value as global `cpu_masks`]
* `workers`: A vector-like config for per-processor worker. For each worker, specify the following fields. System creates 1 worker per device by default and first provided value overrides the settings (i.e., `cpu_masks`, `num_threads`, `profile_copy_computation_ratio`, ... ) and additional field will add additional worker per device.
  * `device`: Target device of specific worker.
    * `CPU`
    * `GPU` 
    * `DSP`
    * `NPU`
  * `cpu_masks`: CPU cluster mask to set CPU affinity of specific worker. [default: same value as global `cpu_masks`]
  * `num_threads`: Number of threads. [default: same value as global `num_threads`]
* `running_time_ms`: Experiment duration in ms. [default: 60000]
* `profile_smoothing_factor`: Current profile reflection ratio. `updated_profile = profile_smoothing_factor * curr_profile + (1 - profile_smoothing_factor) * prev_profile` [default: 0.1]
* `model_profile`: The path to file with model profile results. [default: None]
* `profile_online`: Online profile or offline profile [default: true]
* `profile_warmup_runs`: Number of warmup runs before profile. [default: 1]
* `profile_num_runs`: Number of runs for profile. [default: 1]
* `profile_copy_computation_ratio`: Ratio of computation / input-ouput copy in `list[int]`. Used for latency estimation for each device type (e.g., CPU, GPU, DSP, NPU). The length of the list should be equal to the 4 (`kBandNumDevices`). [default: 30000, 30000, 30000, 30000]
* `schedule_window_size`: The number of planning unit.
* `workload`: The path to file with workload information. [default: None] 


### Example
```
{
    "models": [
        {
            "graph": "/data/local/tmp/model/lite-model_efficientdet_lite0_int8_1.tflite",
            "period_ms": 30,
            "batch_size": 3
        },
        {
            "graph": "/data/local/tmp/model/retinaface_mbv2_quant_160.tflite",
            "period_ms": 30,
            "batch_size": 3
        },
        {
            "graph": "/data/local/tmp/model/ssd_mobilenet_v1_1_metadata_1.tflite",
            "period_ms": 30,
            "batch_size": 3
        }
    ],
    "log_path": "/data/local/tmp/log.csv",
    "schedulers": [
        "heterogeneous_earliest_finish_time_reserved"
    ],
    "minimum_subgraph_size": 7,
    "subgraph_preparation_type": "merge_unit_subgraph",
    "execution_mode": "stream",
    "cpu_masks": "ALL",
    "num_threads": 1,
    "planner_cpu_masks": "PRIMARY",
    "workers": [
        {
            "device": "CPU",
            "num_threads": 1,
            "cpu_masks": "BIG"
        },
        {
            "device": "CPU",
            "num_threads": 1,
            "cpu_masks": "LITTLE"
        },
        {
            "device": "GPU",
            "num_threads": 1,
            "cpu_masks": "ALL"
        },
        {
            "device": "DSP",
            "num_threads": 1,
            "cpu_masks": "PRIMARY"
        },
        {
            "device": "NPU",
            "num_threads": 1,
            "cpu_masks": "PRIMARY"
        }
    ],
    "running_time_ms": 10000,
    "profile_smoothing_factor": 0.1,
    "profile_data_path": "/data/local/tmp/profile.json",
    "profile_online": true,
    "profile_warmup_runs": 3,
    "profile_num_runs": 50,
    "profile_copy_computation_ratio": [
        1000,
        1000,
        1000,
        1000,
        1000
    ],
    "availability_check_interval_ms": 30000,
    "schedule_window_size": 10
}
```

