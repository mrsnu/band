#!/usr/bin/env bash

set -eu

bazel test //tensorflow/lite/testing:interpreter_test --test_output=all
