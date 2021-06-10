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
* `log_path`: The log file path. (e.g., `/data/local/tmp/model_execution_log.csv`)
* `planner`: The planner type in `int`.
  * `0`: Fixed Device Planner
  * `1`: Round-Robin Planner
  * `2`: Shortest Expected Latency Planner
* `execution_mode`: Specify a exeucution mode. Available execution modes are as follows:
  * `stream`: consecutively run batches.
  * `periodic`: invoke requests periodically.
* `cpu_masks`: CPU cluster mask to set CPU affinity. [default: `ALL`]
  * `ALL`: All Cluster
  * `LITTLE`: LITTLE Cluster only
  * `BIG`: Big Cluster only
  * `PRIMARY`: Primary Core only
* `worker_cpu_masks`: CPU cluster mask to set CPU affinity of specific worker. For each worker, specify the mask. [default: `cpu_masks`]
  * `CPU`
  * `CPUFallback`
  * `GPU`
  * `DSP`
  * `NPU`
* `running_time_ms`: Experiment duration in ms. [default: 60000]
* `profile_smoothing_factor`: Current profile reflection ratio. `updated_profile = profile_smoothing_factor * curr_profile + (1 - profile_smoothing_factor) * prev_profile` [default: 0.1]
* `model_profile`: The path to file with model profile results. [default: None]
* `allow_work_steal`: True if work-stealing is allowed. The argument is only effective with `ShortestExpectedLatencyPlanner`.
* `schedule_window_size`: The number of planning unit.

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
            "batch_size": 2
        },
        {
            "graph": "/data/local/tmp/inception_v4.tflite",
            "period_ms": 30,
            "batch_size": 3
        }
    ],
    "log_path": "/data/local/tmp/log.csv",
    "planner": 2,
    "execution_mode": "periodic",
    "cpu_mask": "ALL",
    "worker_cpu_masks": {
      "CPUFallback": "LITTLE",
      "GPU": "BIG"
    },
    "running_time_ms": 60000,
    "profile_smoothing_factor": 0.1,
    "model_profile": "/data/local/tmp/profile.json",
    "allow_work_steal": true,
    "schedule_window_size": 10
}
```

### OPTIONS
Refer to [Benchmark Tool](tensorflow/lite/tools/benchmark) for details.

The following options are added in our version:
* `profile_warmup_runs`: The number of warmup runs during profile stage.
* `profile_num_runs`: The number of iterations during profile stage.
