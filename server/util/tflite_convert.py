import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-t', '--type', type=str, required=True)
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model(args.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

if args.type == "float16":
    converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(args.model + ".tflite", "wb") as f:
    f.write(tflite_model)

