import tensorflow as tf
import os

import dataset


learning_rate = 0.001
num_steps = 10000
batch_size = 128
dropout = 0.75
classes = 10

data = dataset.Dataset(batch_size=batch_size)

train_X, train_Y, train_total = data.train
test_X, test_Y, test_total = data.test


def conv_net(x, n_classes, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        out = tf.layers.dense(fc1, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out


logits_train = conv_net(train_X, classes, dropout, reuse=False, is_training=True)
logits_test = conv_net(test_X, classes, dropout, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=train_Y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

predict = tf.argmax(logits_test, 1)
correct_pred = tf.equal(predict, tf.cast(test_Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(1, num_steps+1):
        _, loss, acc = sess.run([train_op, loss_op, accuracy])
        print('step: ', step, ' loss: ', loss, ' acc: ', acc)
        if acc > 0.98:
            model_name = datetime.now().strftime('%Y%m%d%H%M%S')
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, [])
            with tf.gfile.FastGFile('./pb/{}-{.4f}.pb'.format(model_name, acc), mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            model_name = './ckpt/{}-{.4f}'.format(model_name, acc)
            saver.save(sess, model_name)
    coord.request_stop()
    coord.join(threads)
    print("Done")
