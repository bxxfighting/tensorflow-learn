import cv2 as cv
import os
import tensorflow as tf


class Dataset:
    train_path = '../data/train/images'
    test_path = '../data/test/images'

    def __init__(self, batch_size=128, channels=3):
        self.batch_size = batch_size 
        self.channels = channels

    def __prec_img(self, imgpath):
        img = cv.imread(imgpath)
        img = cv.resize(img, (64, 64))
        return img

    def __read_images(self, dataset_path):
        imagepaths, labels = list(), list()
        classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(int(c))

        total = len(imagepaths)
        imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        image, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)
        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=self.channels)
        image = tf.image.resize_images(image, [64, 64])
        image = image * 1.0/127.5 - 1.0
        x, y = tf.train.batch([image, label], batch_size=self.batch_size,
                capacity=self.batch_size * 8, num_threads=4)

        return x, y, total

    @property
    def test(self):
        return self.__read_images(self.test_path)

    @property
    def train(self):
        return self.__read_images(self.train_path)
