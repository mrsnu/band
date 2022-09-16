TEST_NAME=$1
TARGET=(${TEST_NAME//:/ })

mkdir android_tests
bazel build -c opt --config=android_arm64 $TEST_NAME
cp bazel-bin/${TARGET}/${TARGET[1]} ./android_tests
