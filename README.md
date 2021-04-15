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
* `period_ms`: The delay between subsequent requests in ms. If 0 or below is given, only a few iteraions will run without delay.
* `log_path`: The log file path. (e.g., `/data/local/tmp/model_execution_log.csv`)
* `planner`: The planner type in `int`.
    * `0`: Fixed Device Planner
    * `1`: Round-Robin Planner
    * `2`: Shortest Expected Latency Planner
* `models`: TF Lite models to run. For each model, specify the following fields. 
    * `graph`: TF Lite model path.
    * `batch_size`: The number of model requests in a frame. [default: 1]
    * `device`: Specify the processor to run in int. The argument is only effective with `FixedDevicePlanner`.
* `running_time_ms`: Experiment duration in ms. [default: 60000]
* `model_profile`: The path to file with model profile results. [default: None]
* `cpu_masks`: CPU cluster mask to set CPU affinity. [default: 0]
    * `0`: All Cluster
    * `1`: LITTLE Cluster only
    * `2`: Big Cluster only
    * `3`: Primary Core only
* `execution_mode`: Specify a exeucution mode. Available execution modes are as follows:
    * `stream`: consecutively run batches.
    * `periodic`: invoke requests periodically.
* `schedule_window_size`: The number of planning unit.
* `allow_work_steal`: True if work-stealing is allowed. The argument is only effective with `ShortestExpectedLatencyPlanner`.

An example of complete JSON config file is as follows:
```json
{
    "period_ms": 10,
    "log_path": "/data/local/tmp/log.csv",
    "planner": 2,
    "models": [
        {
          "graph": "/data/local/tmp/mobilenet.tflite",
          "batch_size": 2
        }
    ],
    "running_time_ms": 6000,
    "model_profile": "/data/local/tmp/profile.csv",
    "cpu_masks": 2,
    "execution_mode": "stream",
    "schedule_window_size": 4,
    "allow_work_steal": true
}
```

### OPTIONS
Refer to [Benchmark Tool](tensorflow/lite/tools/benchmark) for details.

The following options are added in our version:
* `profile_warmup_runs`: The number of warmup runs during profile stage.
* `profile_num_runs`: The number of iterations during profile stage.
