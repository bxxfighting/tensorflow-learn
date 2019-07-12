import time
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.compat.v1.enable_eager_execution()

from dataset import dataset


model = tf.keras.models.load_model('./model/model.h5')

test = dataset.test
test_dataset = test['dataset']
test_total = test['total']


batch_size = 1
test_dataset = test_dataset.repeat().shuffle(test_total).batch(batch_size)


for image, label in test_dataset.take(100):
    image = image.numpy()
    label = label.numpy()
    label = label[0]
    predictions = model.predict(image)
    result = np.argmax(predictions[0])
    is_true = False
    if label == result:
        is_true = True
    print(label, result, is_true)
