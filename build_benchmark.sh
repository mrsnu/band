function build_benchmark_binary {
  bazel build -c opt --jobs=`nproc --all` --config=android_arm64 --define tflite_with_xnnpack=true tensorflow/lite/tools/benchmark:benchmark_model
  BUILD_RESULT=$?

  if [ $BUILD_RESULT -eq 0 ]; then
    cp bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model .
    chmod 755 ./benchmark_model
  fi

  return $BUILD_RESULT
}
