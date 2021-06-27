#!/usr/bin/env bash

set -eu

export TENSORFLOW_LIB=tensorflow/tools/libtensorflow_framework.so.2

if [ -f "$TENSORFLOW_LIB" ];then
  echo "Tensorflow library exists."
else
  curl -s -O https://${GITHUBTOKEN}@raw.githubusercontent.com/mrsnu/tflite/tests/${TENSORFLOW_LIB} > ${TENSORFLOW_LIB}
fi
