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

from abc import abstractmethod, ABC

class Model(ABC):
    """This is a parent class that manages all the 
    nodes of the graph including the node itself
    """

    def __init__(self):
        """Initializes all the values for this class
        """

        self.config = None
        self.loss = None
        self.optimizer = None
        self.dataset = None
        self.accuracy = None
        self.dropout = None
        self.graph = None
        self.logits = None
        self._input_placeholder = None
        self._output_placholder = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def compile(self):
        pass
    
    @abstractmethod
    def train(self):
        pass

