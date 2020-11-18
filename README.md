## Default TF-Lite
* Default TF Lite v2.3.0

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
