import time
import os
import math
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import dataset


model = tf.keras.models.load_model('./model/model.h5')

train = dataset.train
train_dataset = train['dataset']
train_total = train['total']
test = dataset.test
test_dataset = test['dataset']
test_total = test['total']


batch_size = 128
train_dataset = train_dataset.repeat().shuffle(train_total).batch(batch_size)
model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(train_total/batch_size))

test_dataset = test_dataset.batch(batch_size)
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_total/batch_size))
print('Accuracy on test dataset:', test_accuracy)

model_name = datetime.now().strftime('%Y%m%d%H%M%S')
model_name = './model/{}.h5'.format(model_name)
model.save(model_name)
