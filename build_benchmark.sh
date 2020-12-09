bazel build -c opt --jobs=16 --config=android_arm64 --define tflite_with_xnnpack=true tensorflow/lite/tools/benchmark:benchmark_model
cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model .
chmod 755 ./benchmark_model
