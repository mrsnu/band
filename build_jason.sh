bazel build -c opt --jobs=`nproc --all` --config=android_arm64 tensorflow/lite/jason:test_jason
cp bazel-bin/tensorflow/lite/jason/test_jason .
chmod 755 test_jason

