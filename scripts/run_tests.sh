#!/usr/bin/env bash

set -eu

source ${PWD}/scripts/get_tensorflow_lib.sh
get_tensorflow_lib
bazel test //tensorflow/lite/testing:tflite_driver_test --test_output=all
delete_tensorflow_lib
