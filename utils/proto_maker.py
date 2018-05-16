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
keras = tf.keras
K = keras.backend
from utils.config import Config
import os.path
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


def proto_maker(model, config):

    if isinstance(config, Config):
        config = config.model
        target_path = os.path.join(config['output_model_path'], "tensorflow")
        name = config['name']
    else:
        raise ValueError('Required : ', Config, 'found : ', type(config))

    try:
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    except IOError as io:
        print(io)
        print('Unable to write the file to given location. Fall back to current director')
        target_path = './tensorflow'
    if isinstance(model, keras.Sequential):
        session = K.get_session()
        K.set_learning_phase(0)
        # remove that learning_phase placeholder to const
        # Because we want our model to have 'output' as the output node name
        # We will use keras's build graph and append to it a identity tf with name='output'
        # afterwards we will freeze the graph.

        with session.graph.as_default():
            # last_layer = tf.get_variable(
            #     name='output', shape=np.shape(model.output))

            tf.identity(model.output, name='output')

        tf.train.write_graph(session.graph_def, target_path, name+'.pbtxt')
        tf.train.Saver().save(session, os.path.join(target_path, name+'ckpt'))
        const_graph = graph_util.convert_variables_to_constants(
            session, session.graph.as_graph_def(), ['output'])
        output_proto = "constant_graph.pb"
        graph_io.write_graph(const_graph, target_path,
                             output_proto, as_text=False)
    else:
        raise ValueError('model should be of type ', type(keras.Sequential))
