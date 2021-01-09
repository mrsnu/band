bazel build -c opt --jobs=`nproc --all` --config=android_arm64 tensorflow/lite/tools/benchmark:benchmark_model
cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model .
chmod 755 ./benchmark_model
