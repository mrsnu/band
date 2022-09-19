#!/bin/bash

MODELS="mobilenetv1 mobilenetv2 mobilenetv3-small mobilenetv3-large resnet50"

mkdir models
python util/keras_models.py --output models
for i in $MODELS
do
	INPUT_DIR=$i
	OUTPUT_DIR="${i}_trt"
	python util/trt_convert.py --input-model-dir models/$INPUT_DIR --output-model-dir models/$OUTPUT_DIR
done
