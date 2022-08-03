import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.experimental.tensorrt import Converter

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-model-dir', type=str, required=True)
parser.add_argument('-o', '--output-model-dir', type=str, required=True)
parser.add_argument('-v', '--version', type=int, default=2)
args = parser.parse_args()

if args.version == 1:
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(args.input_model_dir, "rb") as f:
            frozen_graph = tf.compat.v1.GraphDef()
            frozen_graph.ParseFromString(f.read())

        converter = trt.TrtGraphConverter(
                input_graph_def=frozen_graph)
        converter.convert()
elif args.version == 2:
    converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=args.input_model_dir)
    converter.convert()

converter.save(args.output_model_dir)
