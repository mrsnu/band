import pathlib
import tensorflow as tf

class ImageNetDataset(object):
    def __init__(self, dataset_path):
        self.counter = 0
        self.dataset_path = pathlib.Path(dataset_path)
        self.image_count = len(list(self.dataset_path.glob('*.JPEG')))
        self.list_ds = tf.data.Dataset.list_files(
                str(self.dataset_path/"*.JPEG"), shuffle=False)
        self.image_ds = self.list_ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        self.label_ds = self._load_labels()
        if len(self.image_ds) != len(self.label_ds):
            raise Exception(f"The number of images and that of labels does not match. {len(self.image_ds)} vs. {len(self.label_ds)}")

    def _load_labels(self):
        labels = []
        with open(self.dataset_path/'labels.txt', 'r') as f:
            labels = f.read().splitlines()
        labels = tf.data.Dataset.from_tensor_slices(labels)
        return labels
    
    def _process_path(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [256, 256])
        img = tf.image.resize_with_crop_or_pad(img, 224, 224)
        return img

    def take(self, num=None):
        if num is None:
            return tf.data.Dataset.zip((self.image_ds, self.label_ds))
        return tf.data.Dataset.zip((self.image_ds, self.label_ds)).take(num)

if __name__ == "__main__":
    test_dataset = ImageNetDataset('/home/mjkim/data/val')
    print(test_dataset.take(3))
