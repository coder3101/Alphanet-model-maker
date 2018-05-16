"""
    Copyright 2018 Ashar <ashar786khan@gmail.com>

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import tensorflow as tf
from utils.config import Config
import numpy as np
import os.path
import warnings
from tqdm import trange

Y = 0
X = 0


def build_feed_network(config):
    """Generates a model of feed_forward type

    Arguments:
        config {Config} -- Config file as parsed from YAML

    Raises:
        ValueError -- if config is not a instance of Config

    Returns:
        logits
    """
    if isinstance(config, Config):
        config = config.model
        global X
        X = tf.placeholder(
            tf.float32, shape=[None, config['input_shape']], name='input')

        i_layer = -1
        for L in config['layer_dims'].split(','):
            if i_layer == -1:
                i_layer = tf.layers.dense(X, L, activation=tf.nn.relu)
            else:
                i_layer = tf.layers.dense(i_layer, L, activation=tf.nn.relu)

        logits = tf.layers.dense(
            i_layer, config['output_shape'], activation=tf.nn.softmax, name='output')
    else:
        raise ValueError('required a Config object to build the model')
    return logits


def compile_network(config, logits):
    """Compiles a given model and prepared for training

    Arguments:
        model {keras.Sequencial} -- the model to compile
        config {Config} -- the config file of the model

    Raises:
        ValueError -- if config is not a instance of Config or model is not sequencial

    Returns:
        tf.Graph -- graph with default, optimizer node and loss node
    """
    if isinstance(config, Config):
        config = config.model
        learning_rate = config['learning_rate']
    else:
        raise ValueError('Invalid type, required', Config, 'got', type(config))

    global Y
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(loss)

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32))

    return optimizer, loss, accuracy


def train_network(config, train_x, train_y, validation_x, validation_y, optimizer, loss, accuracy_metric):
    """trains the given compiled network

    Arguments:
        config {Config} -- Config of the given model
        model {keras.Sequencial} -- model to train
        test_x {np.array} -- input_features to train on
        test_y {np.array} -- output_labels to train on
        validation_x -- validation data for features
        validation_y -- validation_data for labels

    Raises:
        ValueError -- if any instance mismatch occurs

    Returns:
        default_graph -- Graph
    """
    if isinstance(config, Config):
        config = config.model
        epoch = config['epoch']
        output_shape = config['output_shape']
        input_shape = config['input_shape']
        output_path = config['output_model_path']
        model_name = config['name']
        batch_size = config['batch_size']
    else:
        raise ValueError('Invalid type, required ',
                         Config, 'found : ', type(config))

    if np.shape(train_x)[1] != input_shape or np.shape(train_y)[1] != output_shape:
        raise TypeError(
            'Input shape and output shape for the model do not fits with config file')

    try:
        target_path = os.path.join(output_path, 'tensorflow')
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    except IOError as io:
        print(io)
        warnings.warn(
            'Skipping file write at {} location shifting to current directory'.format(target_path))
        target_path = './tensorflow'

    # Generate the dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    iter = dataset.shuffle(1000).repeat(epoch).batch(
        batch_size).make_one_shot_iterator()
    next_batch = iter.get_next()
    # Memory Leaks if sess.run(iter.get_next())

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in trange(epoch,desc='Epoch'):
            for inner in range(np.shape(train_x)[0] // batch_size):
                batch_x, batch_y = sess.run(next_batch)
                if inner % 100 == 0:
                    print('Loss at epoch {} and iteration : {} is {}. '.format(i, inner, sess.run(
                        loss, feed_dict={X: batch_x, Y: batch_y})))
                if inner % 200 == 0:
                    print('Accuracy at epoch {} and iteration : {} is {}. '.format(i, inner, sess.run(
                        accuracy_metric, feed_dict={X: batch_x, Y: batch_y})))
                sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            # Save the learning after each check points
        tf.train.write_graph(sess.graph_def, target_path, model_name+'.pbtxt')
        saver.save(sess, os.path.join(target_path, model_name+'ckpt'))
        print('\n\nSaved a model with accuracy of {} on validation data.'.format(
            sess.run(accuracy_metric, feed_dict={X: validation_x, Y: validation_y})))

    return tf.get_default_graph()
