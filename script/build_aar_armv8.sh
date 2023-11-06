rm -f band.aar band_dbg.aar
bazel build -c opt --config android_arm64_tflite --define tflite_with_xnnpack=false //band/java:band
cp bazel-bin/band/java/band.aar ./
bazel build -c dbg --config android_arm64_tflite --define tflite_with_xnnpack=false //band/java:band
cp bazel-bin/band/java/band.aar ./band_dbg.aar

