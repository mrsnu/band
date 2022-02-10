# Build for android
bazel build -c opt --jobs=`nproc --all` --config=android_arm64 //tensorflow/lite/c:tensorflowlite_c
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.so .

# Build for ios, linux, windows, ...
bazel build -c opt --jobs=`nproc --all` //tensorflow/lite/c:tensorflowlite_c
cp bazel-bin/tensorflow/lite/c/libtensorflowlite_c.dylib .
cp bazel-bin/tensorflow/lite/c/tensorflowlite_c.dll .
