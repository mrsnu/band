bazel build -c opt --verbose_failures --fat_apk_cpu=arm64-v8a   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain   //tensorflow/lite/java:tensorflow-lite
cp bazel-bin/tensorflow/lite/java/tensorflow-lite.aar .
