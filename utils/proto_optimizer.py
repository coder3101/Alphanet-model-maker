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

from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference_lib
import tensorflow as tf
keras = tf.keras
K = keras.backend
from utils.config import Config
import os.path


def proto_optimizer(config, model):
    if isinstance(config, Config):
        config = config.model
        name = config.name
        target_path = os.path.join(config.output_model_path)
    else:
        raise ValueError('required : ', Config, 'found : ', type(config))

    if isinstance(model, keras.Sequential):
        graph = tf.GraphDef()
        with tf.gfile.FastGFile(os.path.join(target_path, 'tensorflow/frozen_'+name+'.pb'), 'rb') as f:
            data = f.read()
            graph.ParseFromString(data)
        output_graph_def = optimize_for_inference_lib(
            input_graph_def=graph,
            input_node_names=['input'],
            output_node_name=['output'],
            placeholder_type_enum=tf.float32.as_datatype_enum)

        f2 = tf.gfile.FastGFile(os.path.join(
            target_path, "optimized" + name + ".pb"), 'w')
        f2.write(output_graph_def.SerializeToString())
        f2.close()

    else:
        raise ValueError('sequencial model type required found', type(model))
