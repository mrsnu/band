import pathlib
import argparse
import tensorflow as tf
import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-p', '--partition', type=int, required=True)
args = parser.parse_args()

partition_points = {
        'mobilenetv2': {
            'input': ["input_2", "block_15_add"],
            'output': "predictions",
        },
        'resnet50': {
            'input': ["input_5"],
            "output": "predictions",
        }
    }

def partition_model(model, partition_point):
    input_layer = model.get_layer(partition_point['input'][0]).output
    imm_layer = model.get_layer(partition_point['input'][args.partition]).output
    output_layer = model.get_layer(partition_point['output']).output
    model_first = keras.Model(inputs=input_layer, outputs=imm_layer)
    model_second = keras.Model(inputs=imm_layer, outputs=output_layer)
    return model_first, model_second

def load_model(model_path):
    return keras.models.load_model(model_path)

def main():
    model_name = pathlib.Path(args.model).name
    model = load_model(args.model)
    model_1, model_2 = partition_model(model, partition_points[model_name])
    model_1.summary()
    model_2.summary()
    model_1.save(args.model + '_part0')
    model_2.save(args.model + '_part1')

if __name__ == "__main__":
    main()
