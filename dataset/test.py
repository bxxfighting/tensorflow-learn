import time
import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from dataset import dataset


model = tf.keras.models.load_model('./model/model.h5')

test = dataset.test
test_dataset = test['dataset']
test_total = test['total']


batch_size = 128
test_dataset = test_dataset.batch(batch_size)
model.evaluate(test_dataset)
