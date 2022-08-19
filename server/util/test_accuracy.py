import pathlib
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import *
from tensorflow import keras
from tqdm import tqdm
from PIL import Image

from dataloader import ImageNetDataset

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='/home/mjkim/data/val', required=False)
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-p', '--partition', type=int, default=0, required=False)
args = parser.parse_args()

keras_package = {
        'mobilenetv1': mobilenet,
        'mobilenetv1_trt': mobilenet,
        'mobilenetv2': mobilenet_v2,
        'mobilenetv2_trt': mobilenet_v2,
        'resnet50': resnet50,
        'resnet50_trt': resnet50
    }

partition_points = {
        'mobilenetv2': {
            'input': ["input_2", "block_15_add"],
            'output': "predictions",
        },
        'resnet50': {
            'input': [],
            "output": "",
        }
    }

def partition_model(model, partition_point):
    input_layer = model.get_layer(partition_point['input'][args.partition]).output
    output_layer = model.get_layer(partition_point['output']).output
    return keras.Model(inputs=input_layer, outputs=output_layer)

def load_model(model_path):
    return keras.models.load_model(model_path)

def main():
    model_name = pathlib.Path(args.model).name
    dataset = ImageNetDataset(args.dataset).take().batch(128)

    model = load_model(args.model)
    model = partition_model(model, partition_points[model_name])
    
    count = 0
    preprocessor = keras_package[model_name].preprocess_input
    postprocessor = keras_package[model_name].decode_predictions
    for i, (img, labels) in enumerate(tqdm(dataset)):
        pred = model(preprocessor(img))
        pred = postprocessor(pred.numpy(), top=1)
        pred = np.array(list(map(lambda x: bytes(x[0][0], 'utf-8'), pred)))
        count += np.count_nonzero(pred == labels.numpy())
    print(f"Accuracy: {count / 50_000 * 100}%")

if __name__ == "__main__":
    main()
