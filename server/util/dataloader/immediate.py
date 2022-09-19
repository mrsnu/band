import json
import pathlib
import numpy as np
import tensorflow as tf
from .imagenet import ImageNetDataset

class ImageNetImmDataset(ImageNetDataset):
    def __init__(self, dataset_path):
        self.counter = 0
        self.dataset_path = pathlib.Path(dataset_path)
        self.image_count = len(list(self.dataset_path.glob('*.json')))
        self.list_ds = tf.data.Dataset.list_files(
                str(self.dataset_path/"*.json"), shuffle=False)
        self.image_ds = self.list_ds.map(
                lambda x: tf.py_function(self._process_path, [x], [tf.float32]), 
                num_parallel_calls=tf.data.AUTOTUNE)
        self.label_ds = self._load_labels()
        if len(self.image_ds) != len(self.label_ds):
            raise Exception(f"The number of images and that of labels does not match. {len(self.image_ds)} vs. {len(self.label_ds)}")

    def _process_path(self, file_path):
        img = tf.io.read_file(file_path)
        img = json.loads(img.numpy())
        shape = img['shape']
        img = np.float32(img['data']).reshape(shape[1:])
        return tf.convert_to_tensor(img)

    def take(self, num=None):
        if num is None:
            return tf.data.Dataset.zip((self.image_ds, self.label_ds))
        return tf.data.Dataset.zip((self.image_ds, self.label_ds)).take(num)

if __name__ == "__main__":
    test_dataset = ImageNetImmDataset('/home/mjkim/data/imm')
    for k, v in test_dataset.take(1):
        print(k)
        print(v)
