#!/usr/bin/env bash

set -eu

export TENSORFLOW_LIB=libtensorflow_framework.so.2
export TENSORFLOW_LIB_PATH=tensorflow/${TENSORFLOW_LIB}

if [ -f "$TENSORFLOW_LIB_PATH" ];then
  echo "Tensorflow library exists."
else
  curl -s -O https://${GITHUBTOKEN}@raw.githubusercontent.com/mrsnu/tflite/tests/${TENSORFLOW_LIB_PATH} && mv ${TENSORFLOW_LIB} ${TENSORFLOW_LIB_PATH}
fi
