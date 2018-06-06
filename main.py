import warnings

import numpy as np
import tensorflow as tf
import yaml
from utils.config import Config
from utils.dataset import Dataset
from utils.YAMLvalidator import validate_config

from model import feed_forward
from model.feed_forward import FeedForward

try:
    with open('config.yaml') as f:
        dataMap = yaml.safe_load(f)
except Exception as e:
    print(e)
    exit(-1)

validate_config(dataMap)
config = Config(**dataMap)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Could be replaced with tf.one_hot
# For Data preprocessing fall back to keras API's for the moment.
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_test = np.reshape(x_test, (-1, 784))
x_train = np.reshape(x_train, (-1, 784))

x_train = x_train / 255
x_test = x_test / 255

dataSet = Dataset()
dataSet.set_test_data((x_test, y_test))
dataSet.set_train_data((x_train, y_train))
dataSet.prepare_data(shuffle_all=True)

feed_forward = FeedForward(config,dataSet)
feed_forward.build()
feed_forward.compile()
feed_forward.train()

# from utils.proto_maker import proto_maker
# proto_maker(model,config)
