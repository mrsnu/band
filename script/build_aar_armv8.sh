rm -f band.aar
bazel build -c dbg --config android_arm64_tflite --define tflite_with_xnnpack=false //band/java:band
cp bazel-bin/band/java/band.aar ./

