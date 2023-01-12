rm -f band.aar
bazel build -c dbg --config android_arm64_tflite //band/java:band
cp bazel-bin/band/java/band.aar ./

