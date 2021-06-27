#!/usr/bin/env bash

set -eu

bazel test //tensorflow/lite/testing:tflite_driver_test --test_output=all
