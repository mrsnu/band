## Codebase
* Tensorflow Lite in [Tensorflow v2.3.0](https://github.com/tensorflow/tensorflow/tree/v2.3.0) 
* Among files and directories located in `[root]/tensorflow`, files unrelated to TF Lite are deleted.
  * The following directories are necessary to build `benchmark tool` and `Android library`.
  * Remaining directories in `[root]/tensorflow/` are as follows:
    ```
    [root]/tensorflow
    ├── lite
    │   └── <Same as TF v2.3.0>
    ├── java
    │   └── <Same as TF v2.3.0>
    ├── core
    │   ├── kernels
    │   ├── platform
    │   ├── public
    │   └── util
    └── tools
        ├── def_file_filter
        └── git
    ```

## How To Build
* tested with [bazel 3.1.0](https://github.com/bazelbuild/bazel/releases/tag/3.1.0)
* install the submodule ([Ruy](https://github.com/mrsnu/ruy/tree/tf_v2.3.0))
```bash
git submodule update --init --recursive
```
### Benchmark Tool
`Benchmark Tool` enables op-level profiling.

`[root]/build_benchmark.sh` script will generate `Benchmark Tool` binary file in the root directory.

More details on how to use the tool can be found in the [documentation](https://github.com/mrsnu/tflite/tree/master/tensorflow/lite/tools/benchmark).

### Android Library
One example of building `Android Library` can be found in `[root]/build_aar_armv8.sh`.

## Benchmark Tool Usage
Run multi-DNN experiments with our modified version of [TF Lite Benchmark Tool](tensorflow/lite/tools/benchmark).

### Basic Usage
Run binary file with the `--json_path` option. For example:
```bash
$ adb shell /data/local/tmp/benchmark_model --json_path=$PATH_TO_CONFIG_FILE [OPTIONS]
```

### JSON Config file arguments
* `models`: TF Lite models to run. For each model, specify the following fields.
  * `graph`: TF Lite model path.
  * `period_ms`: The delay between subsequent requests in ms.
  * `batch_size`: The number of model requests in a frame. [default: 1]
  * `device`: Specify the processor to run in int. The argument is only effective with `FixedDevicePlanner`.
  * `input_layer`: Input layer names.
  * `input_layer_shape`: Input layer shape.
  * `input_layer_value_range`: A map-like string representing value range for *integer* input layers. Each item is separated by ':', and the item value consists of input layer name and integer-only range values (both low and high are inclusive) separated by ',', e.g. input1,1,2:input2,0,254.
  * `input_layer_value_files`: A map-like string representing value file. Each item is separated by ',', and the item value consists of input layer name and value file path separated by ':', e.g. input1:file_path1,input2:file_path2. If the input_name appears both in input_layer_value_range and input_layer_value_files, input_layer_value_range of the input_name will be ignored. The file format is binary and it should be array format or null separated strings format.
  * `slo_us` and `slo_scale`: **Optional** fields for specifying an SLO value for a model. Setting `slo_scale` will make the SLO = worst profiled latency of that model * `slo_scale`. `slo_scale` will be ignored if `slo_us` is given (i.e., no reason to specify both options).
* `log_path`: The log file path. (e.g., `/data/local/tmp/model_execution_log.csv`)
* `schedulers`: The scheduler types in `list[int]`. If N schedulers are specified, then N queues are generated.
  * `0`: Fixed Device Planner
  * `1`: Round-Robin Planner
  * `2`: Shortest Expected Latency Planner
* `minimum_subgraph_size`: Minimum subgraph size. If candidate subgraph size is smaller than `minimum_subgraph_size`, the subgraph will not be created. [default: 7]
* `subgraph_preparation_type`: For schedulers using fallback, determine how to generate candidate subgraphs. [default: `merge_unit_subgraph`]
  * `no_fallback_subgraph`: Generate subgraphs per device. Explicit fallback subgraph will not be generated.
  * `fallback_per_device`: Generate fallback subgraphs for each device.
  * `unit_subgraph`: Generate unit subgraphs considering all device supportiveness. All ops in same unit subgraph have same support devices.
  * `merge_unit_subgraph`: Add merged unit subgraphs to `unit_subgraph`.
* `execution_mode`: Specify a exeucution mode. Available execution modes are as follows:
  * `stream`: consecutively run batches.
  * `periodic`: invoke requests periodically.
  * `periodic_single_thread`: invoke requests periodically with a single thread.
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
    * `CPUFallback`
    * `GPU`
    * `DSP`
    * `NPU`
  * `cpu_masks`: CPU cluster mask to set CPU affinity of specific worker. [default: same value as global `cpu_masks`]
  * `num_threads`: Number of threads. [default: same value as global `num_threads`]
* `disabled_devices`: List of devices to disable. [default: None]
* `running_time_ms`: Experiment duration in ms. [default: 60000]
* `profile_smoothing_factor`: Current profile reflection ratio. `updated_profile = profile_smoothing_factor * curr_profile + (1 - profile_smoothing_factor) * prev_profile` [default: 0.1]
* `model_profile`: The path to file with model profile results. [default: None]
* `profile_online`: Online profile or offline profile [default: true]
* `profile_warmup_runs`: Number of warmup runs before profile. [default: 1]
* `profile_num_runs`: Number of runs for profile. [default: 1]
* `profile_copy_computation_ratio`: Ratio of computation / input-ouput copy. Used for latency estimation. [default: 1000]
* `allow_work_steal`: True if work-stealing is allowed. The argument is only effective with `ShortestExpectedLatencyPlanner`.
* `availability_check_interval_ms`: Availability check interval for disabled device due to thermal throttling. [default: 30000]
* `schedule_window_size`: The number of planning unit.
* `global_period_ms`: Request interval value used for execution mode `periodic_single_thread` only. Should be > 0.
* `model_id_random_seed`: Random seed value used for picking model ids, in `periodic_single_thread` only. 0 is treated as the current timestamp.
* `workload`: The path to file with workload information. [default: None] 

An example of complete JSON config file is as follows:
```json
{
    "models": [
        {
            "graph": "/data/local/tmp/mobilenet.tflite",
            "period_ms": 10,
            "batch_size": 1,
            "input_layer" : "input",
            "input_layer_shape" : "1,224,224,3",
            "input_layer_value_range" : "input,1,3"
        },
        {
            "graph": "/data/local/tmp/yolov3.tflite",
            "period_ms": 20,
            "batch_size": 2,
            "slo_scale": 2.0
        },
        {
            "graph": "/data/local/tmp/inception_v4.tflite",
            "period_ms": 30,
            "batch_size": 3,
            "slo_us": 30000
        }
    ],
    "log_path": "/data/local/tmp/log.csv",
    "schedulers": [0, 2],
    "minimum_subgraph_size": 7,
    "subgraph_preparation_type": "merge_unit_subgraph",
    "execution_mode": "periodic",
    "cpu_masks": "ALL",
    "num_threads": 1,
    "planner_cpu_masks": "PRIMARY",
    "workers": [
      {
        "device": "CPU",
        "num_threads": 3,
        "cpu_masks": "BIG",
        "profile_copy_computation_ratio": 10
      },
      {
        "device": "CPU",
        "num_threads": 4,
        "cpu_masks": "LITTLE",
        "profile_copy_computation_ratio": 10
      }
    ],
    "disabled_devices": ["GPU"],
    "running_time_ms": 60000,
    "profile_smoothing_factor": 0.1,
    "model_profile": "/data/local/tmp/profile.json",
    "profile_online": true,
    "profile_warmup_runs": 1,
    "profile_num_runs": 1,
    "profile_copy_computation_ratio": 1000,
    "allow_work_steal": true,
    "availability_check_interval_ms": 30000,
    "schedule_window_size": 10
}
```

### OPTIONS
Refer to [Benchmark Tool](tensorflow/lite/tools/benchmark) for details.
