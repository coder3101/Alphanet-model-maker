import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_test = np.reshape(x_test, (-1, 784))/255
x_train = np.reshape(x_train, (-1, 784))/255

X = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

h1 = tf.get_variable('h1',shape=[784, 20])
b1 = tf.zeros(shape=[20])

h2 = tf.get_variable('h2', shape=[20,10])
b2 = tf.zeros(shape=[10])

l1 = tf.nn.relu(tf.matmul(X, h1) + b1)
logits = tf.matmul(l1, h2) + b2

# l1 = tf.layers.dense(inputs=X, units=16, activation=tf.nn.relu)
# logits = tf.layers.dense(inputs=l1, units=10, activation=tf.nn.softmax)

loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32))

data_graph = tf.Graph()
with data_graph.as_default():
    data_pipeline = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    iter = data_pipeline.shuffle(1000).repeat().batch(64).make_one_shot_iterator()
    next_item = iter.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        for i2 in range(100):
            sess2 = tf.Session(graph=data_graph)
            x, y = sess2.run(next_item)
            sess2.close()
            if i2 % 100 == 0:
                lss = sess.run(loss, feed_dict={X: x, Y: y})
                print('Loss at {} => {} is {}'.format(i, i2, lss))
                sess.run(optimizer, feed_dict={X: x, Y: y})
    print('Final Accuracy : ', sess.run(
        accuracy, feed_dict={X: x_train, Y: y_train}))
    #tf.train.write_graph(sess.graph, './', 'temp.pbtxt')
