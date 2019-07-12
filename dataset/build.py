import time
import os
import math
import numpy as np
import tensorflow as tf
from dataset import dataset


train = dataset.train
train_dataset = train['dataset']
train_total = train['total']
test = dataset.test
test_dataset = test['dataset']
test_total = test['total']


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu, input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 128
train_dataset = train_dataset.repeat().shuffle(train_total).batch(batch_size)
model.fit(train_dataset, epochs=20, steps_per_epoch=math.ceil(train_total/batch_size))

test_dataset = test_dataset.batch(batch_size)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_total/batch_size))
print('Accuracy on test dataset:', test_accuracy)

model_name = datetime.now().strftime('%Y%m%d%H%M%S')
model_name = './model/{}-{}.h5'.format(model_name, test_accuracy)
model.save(model_name)

