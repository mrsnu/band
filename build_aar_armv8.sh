bazel build -c opt --jobs=8 --verbose_failures --fat_apk_cpu=arm64-v8a   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain   //tensorflow/lite/java:tensorflow-lite //tensorflow/lite/java:tensorflow-lite-gpu
cp bazel-bin/tensorflow/lite/java/tensorflow-lite.aar .
cp bazel-bin/tensorflow/lite/java/tensorflow-lite-gpu.aar .