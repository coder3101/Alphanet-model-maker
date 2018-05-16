# import tensorflow as tf
# import numpy as np

# # load the dataset from keras api
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
# x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
# x_train = x_train / 255
# x_test = x_test / 255

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.InputLayer((28, 28, 1), batch_size=32))
# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D())
# model.add(tf.keras.layers.Dropout(0.25))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
# model.compile(loss=tf.keras.losses.categorical_crossentropy,
#               optimizer=tf.keras.optimizers.Adagrad(),
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=32,
#           epochs=5,
#           verbose=1,
#           validation_data=(x_test, y_test))

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# model.save('MNIST_CNN.h5')


import yaml
import warnings
from utils.YAMLvalidator import validate_config
from model import feed_forward
from utils.config import Config

try:
    with open('config.yaml') as f:
        dataMap = yaml.safe_load(f)
except Exception as e:
    print(e)
    exit(-1)

validate_config(dataMap)

config = Config(**dataMap)

logits = feed_forward.build_feed_network(config)

optimizer, loss, accr = feed_forward.compile_network(config, logits)

import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Could be replaced with tf.one_hot
# For Data preprocessing fall back to keras API's for the moment.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_test = np.reshape(x_test,(-1,784))
x_train = np.reshape(x_train,(-1,784))

# y_train = np.reshape(y_train, (10,-1))
# y_test = np.reshape(y_test, (10,-1))

x_train = x_train / 255
x_test = x_test / 2


feed_forward.train_network(config, x_train, y_train, x_test, y_test, optimizer, loss, accr)

# from utils.proto_maker import proto_maker
# proto_maker(model,config)