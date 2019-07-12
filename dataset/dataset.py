import time
import os
import math
import numpy as np
import tensorflow as tf


class Dataset:
    train_dataset_path = './data/images/'
    test_dataset_path = './data/test/images/'

    def read_images(self, dataset_path):
        label = 0
        imagepaths = []
        labels = []
        classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
        return imagepaths, labels


    def pre_proc(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [64, 64])
        return image, label

    @property
    def test(self):
        images, labels = self.read_images(self.test_dataset_path)
        total = len(images)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self.pre_proc)
        data = {
            'dataset': dataset,
            'total': total,
        }
        return data

    @property
    def train(self):
        images, labels = self.read_images(self.train_dataset_path)
        total = len(images)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self.pre_proc)
        data = {
            'dataset': dataset,
            'total': total,
        }
        return data

dataset = Dataset()
