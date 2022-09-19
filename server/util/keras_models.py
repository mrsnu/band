import os
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import (
        MobileNetV3Small,
        MobileNetV3Large
    )
from tensorflow.keras.applications import ResNet50

models = {
        "mobilenetv1": MobileNet,
        "mobilenetv2": MobileNetV2,
        "mobilenetv3-small": MobileNetV3Small,
        "mobilenetv3-large": MobileNetV3Large,
        "resnet50": ResNet50
        }

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="all", required=False)
parser.add_argument("-o", "--output", type=str, default="", required=False)
args = parser.parse_args()
if args.model not in models and args.model != "all":
    raise ValueError("Unsupported model.")

if args.model == "all":
    for k, v in models.items():
        model = v()
        model.save(os.path.join(args.output, k))
else:
    model = models[args.model]()
    model.save(os.path.join(args.output, args.model))
