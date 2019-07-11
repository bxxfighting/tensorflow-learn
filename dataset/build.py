'''
在学习tensorflow的时候，开始总是用mnist数据集，而我现在自己有一些照片，我想训练自己的数据
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/5_DataManagement/build_an_image_dataset.ipynb
'''
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def read_images(dataset_path):
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


def pre_proc(filename, label):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [64, 64])
    return image, label

dataset_path = 'data/images/'
images, labels = read_images(dataset_path)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3, random_state=0)
train_total = len(train_images)
test_total = len(test_images)
print(train_total, test_total)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(pre_proc)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.train.AdamOptimizer(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
train_dataset = train_dataset.repeat().shuffle(train_total).batch(batch_size)
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(train_total/batch_size))


test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.map(pre_proc)
test_dataset = test_dataset.batch(batch_size)
test_loss, test_accuracy = model.evaluate(train_dataset, steps=math.ceil(train_total/batch_size))
print('Accuracy on test dataset:', test_accuracy)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_total/batch_size))
print('Accuracy on test dataset:', test_accuracy)
