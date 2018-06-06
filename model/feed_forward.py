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

import os.path
import warnings

import numpy as np
import tensorflow as tf
from tqdm import trange
from utils.config import Config
from utils.dataset import Dataset

# Y = 0
# X = 0


# def build_feed_network(config):
#     """Generates a model of feed_forward type

#     Arguments:
#         config {Config} -- Config file as parsed from YAML

#     Raises:
#         ValueError -- if config is not a instance of Config

#     Returns:
#         logits
#     """
#     if isinstance(config, Config):
#         config = config.model  # pylint: disable=CODE
#         global X
#         X = tf.placeholder(
#             tf.float32, shape=[None, config['input_shape']], name='input')

#         i_layer = -1
#         for layer_size in config['layer_dims'].split(','):
#             if i_layer == -1:
#                 i_layer = tf.layers.dense(
#                     X, layer_size, activation=tf.nn.leaky_relu)
#             else:
#                 i_layer = tf.layers.dense(
#                     i_layer, layer_size, activation=tf.nn.leaky_relu)

#         logits = tf.layers.dense(
#             i_layer, config['output_shape'], activation=tf.nn.softmax, name='output')
#     else:
#         raise ValueError('required a Config object to build the model')
#     return logits


# def compile_network(config, logits):
#     """Compiles a given model and prepared for training

#     Arguments:
#         model {keras.Sequencial} -- the model to compile
#         config {Config} -- the config file of the model

#     Raises:
#         ValueError -- if config is not a instance of Config or model is not sequencial

#     Returns:
#         tf.Graph -- graph with default, optimizer node and loss node
#     """
#     if isinstance(config, Config):
#         config = config.model
#         learning_rate = config['learning_rate']
#     else:
#         raise ValueError('Invalid type, required', Config, 'got', type(config))

#     global Y
#     Y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

#     loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
#     optimizer = tf.train.AdamOptimizer(
#         learning_rate=learning_rate).minimize(loss)

#     accuracy = tf.reduce_mean(
#         tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(Y, -1)), tf.float32))

#     return optimizer, loss, accuracy


# def train_network(config, dataset, optimizer, loss, accuracy_metric):
#     """trains the given compiled network

#     Arguments:
#         config {Config} -- Config of the given model
#         dataset {Dataset} -- Dataset class having the data into it


#     Raises:
#         ValueError -- if any instance mismatch occurs

#     Returns:
#         default_graph -- Graph
#     """
#     if isinstance(config, Config):
#         config = config.model
#         epoch = config['epoch']
#         output_shape = config['output_shape']
#         input_shape = config['input_shape']
#         output_path = config['output_model_path']
#         model_name = config['name']
#         batch_size = config['batch_size']
#     else:
#         raise ValueError('Invalid type, required ',
#                          Config, 'found : ', type(config))

#     if np.shape(dataset._train_data[0])[1] != input_shape or np.shape(dataset._train_data[1])[1] != output_shape:
#         raise TypeError(
#             'Input shape and output shape for the model do not fits with config file')

#     try:
#         target_path = os.path.join(output_path, 'tensorflow')
#         if not os.path.exists(target_path):
#             os.mkdir(target_path)
#     except IOError as io:
#         print(io)
#         warnings.warn(
#             'Skipping file write at {} location shifting to current directory'.format(target_path))
#         target_path = './tensorflow'

#     # Generate the dataset pipeline
#     # fixme(coder3101) : This segment is causing memory exhaustion
#     # this pipeline should be used with only GPU or large memory devices.
# # =============================================================================
# # | dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))           |
# # | iter = dataset.shuffle(1000).repeat(epoch).batch(                          |
# # |    batch_size).make_one_shot_iterator()                                    |
# # | next_batch = iter.get_next()                                               |
# # ==============================================================================
#     # Memory Leaks if sess.run(iter.get_next())
#     # Falling back to naive approach for shuffled and batched data output

#     saver = tf.train.Saver()

#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for i in range(epoch):
#             inner = trange(np.shape(dataset._train_data[0])[0] // batch_size)
#             for j in inner:
#                 batch_x, batch_y = dataset.next_batch_from_train(batch_size)
#                 pp, ss = dataset.next_batch_from_test(batch_size)
#                 li, ai = sess.run([loss, accuracy_metric],
#                                   feed_dict={X: pp, Y: ss})
#                 inner.set_postfix(loss=li, accuracy=ai)
#                 inner.set_description('Epoch ({}/{})'.format(i+1, epoch))
#                 sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

#         tf.train.write_graph(sess.graph_def, target_path, model_name+'.pbtxt')
#         saver.save(sess, os.path.join(target_path, model_name+'ckpt'))
#         print('\n\nSaved a model with accuracy of {} on validation data.'.format(
#             sess.run(accuracy_metric, feed_dict={X: dataset._test_data[0], Y: dataset._test_data[1]})))

#     return tf.get_default_graph()


from model.model import Model


class FeedForward(Model):
    def __init__(self, config, dataset):
        super().__init__()
        if isinstance(config, Config):
            self.config = config
        else:
            raise ValueError(
                'Required a config class instance got ', type(config))
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise ValueError(
                'Required a dataset as second argument. Got', type(dataset))

    def build(self):
        config = self.config.model
        self._input_placeholder = tf.placeholder(
            tf.float32, shape=[None, config['input_shape']], name='input')

        i_layer = -1
        for layer_size in config['layer_dims'].split(','):
            if i_layer == -1:
                i_layer = tf.layers.dense(
                    self._input_placeholder, layer_size, activation=tf.nn.leaky_relu)
            else:
                i_layer = tf.layers.dense(
                    i_layer, layer_size, activation=tf.nn.leaky_relu)

        self.logits = tf.layers.dense(
            i_layer, config['output_shape'], activation=tf.nn.softmax, name='output')

    def compile(self):
        config = self.config.model
        logits = self.logits
        learning_rate = config['learning_rate']
        self._output_placholder = tf.placeholder(
            dtype=tf.float32, shape=[None, 10])
        self.loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self._output_placholder, logits=logits)
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, -1), tf.argmax(self._output_placholder, -1)), tf.float32))
        self.graph = tf.get_default_graph()

    def train(self):
        config = self.config.model
        dataset = self.dataset
        epoch = config['epoch']
        output_shape = config['output_shape']
        input_shape = config['input_shape']
        output_path = config['output_model_path']
        model_name = config['name']
        batch_size = config['batch_size']

        if np.shape(dataset._train_data[0])[1] != input_shape or np.shape(dataset._train_data[1])[1] != output_shape:
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
        # fixme(coder3101) : This segment is causing memory exhaustion
        # this pipeline should be used with only GPU or large memory devices.
    # =============================================================================
    # | dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))           |
    # | iter = dataset.shuffle(1000).repeat(epoch).batch(                          |
    # |    batch_size).make_one_shot_iterator()                                    |
    # | next_batch = iter.get_next()                                               |
    # ==============================================================================
        # Memory Leaks if sess.run(iter.get_next())
        # Falling back to naive approach for shuffled and batched data output

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                inner = trange(np.shape(dataset._train_data[0])[
                               0] // batch_size)
                for j in inner:
                    batch_x, batch_y = dataset.next_batch_from_train(
                        batch_size)
                    pp, ss = dataset.next_batch_from_test(batch_size)
                    li, ai = sess.run([self.loss, self.accuracy],
                                      feed_dict={self._input_placeholder: pp, self._output_placholder: ss})
                    inner.set_postfix(loss=li, accuracy=ai)
                    inner.set_description('Epoch ({}/{})'.format(i+1, epoch))
                    sess.run(self.optimizer, feed_dict={
                             self._input_placeholder: batch_x, self._output_placholder: batch_y})

            tf.train.write_graph(
                sess.graph_def, target_path, model_name+'.pbtxt')
            saver.save(sess, os.path.join(target_path, model_name+'ckpt'))
            print('\n\nSaved a model with accuracy of {} on validation data.'.format(
                sess.run(self.accuracy, feed_dict={self._input_placeholder: dataset._test_data[0], self._output_placholder: dataset._test_data[1]})))
