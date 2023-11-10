rm -f splash*benchmark*.run

bazel build -c opt --config android_arm64_tflite --config trace --config splash //band/tool:band_benchmark
cp bazel-bin/band/tool/band_benchmark ./splash_benchmark.run
bazel build -c dbg --config android_arm64_tflite --config trace --config splash //band/tool:band_benchmark
cp bazel-bin/band/tool/band_benchmark ./splash_benchmark_dbg.run