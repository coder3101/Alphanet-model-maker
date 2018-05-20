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
import warnings
import math
import numpy as np


class Dataset:
    """The dataset class that we will use incase we are training on lower end devices
    those which cause tf.data.Dataset pipeline to freeze. This is alternative to it.

    Raises:
        ValueError -- Incorrect values to parameters
        ValueError -- Incorrect values to parameters
        RuntimeError -- Test batch exceeds 1 epoch
        ValueError -- Incorrect value to parameters

    Returns:
        [Dataset] -- A dataset object
    """

    def __init__(self, total_data=None, split_ratio=None, batch=0, epoch=1):
        self._full_data = total_data
        self._test_data = None
        self._train_data = None
        self._last_batch_test = 0
        self._last_batch_train = 0
        self._epoch_count = 0
        self.epoch = epoch
        self._last_test_batch_shown = False
        self.batch_size = batch
        if split_ratio != None and total_data != None:
            self._test_data, self._train_data = self._split_data(split_ratio)

    def _split_data(self, test_train_split_ratio=0.2):
        self._shuffle_all()
        total_size = len(self._full_data[0])
        test_x = test_y = train_x = train_y = None

        if test_train_split_ratio > 1:
            raise ValueError('test_train_split_ratio cannot be more than 1')

        if test_train_split_ratio >= 0.8:
            warnings.warn('Very high test train ratio for split.')
        test_count = test_train_split_ratio*total_size

        test_x = self._full_data[0][0:test_count]
        test_y = self._full_data[1][0:test_count]
        train_x = self._full_data[0][test_count:]
        train_y = self._full_data[1][test_count:]

        return (test_x, test_y), (train_x, train_y)

    def _shuffle_all(self):
        x, y = self._full_data
        assert len(x) == len(y)
        t = np.random.permutation(len(x))
        self._full_data = x[t], y[t]

    def set_train_data(self, train_data, shuffle=False):
        """sets the train data
        
        Arguments:
            train_data {np.array} -- Train data
        
        Keyword Arguments:
            shuffle {bool} -- wheather to shuffle this dataset (default: {False})
        """

        if shuffle:
            x, y = train_data
            assert len(x) == len(y)
            t = np.random.permutation(len(x))
            self._train_data = x[t], y[t]
        else:
            self._train_data = train_data

    def set_test_data(self, test_data, shuffle=False):
        """sets the test data
        
        Arguments:
            test_data {np.array} -- Test data for the dataset
        
        Keyword Arguments:
            shuffle {bool} -- shoould we shuffle this data before saving (default: {False})
        """

        if shuffle:
            x, y = test_data
            assert len(x) == len(y)
            t = np.random.permutation(len(x))
            self._test_data = x[t], y[t]
        else:
            self._test_data = test_data

    def prepare_data(self, shuffle_all=True):
        """Preapares the dataset for access

        Keyword Arguments:
            shuffle_all {bool} -- If to shuffle the test and train before serving (default: {True})

        """
        if shuffle_all:
            x, y = self._test_data
            assert len(x) == len(y)
            x1, y1 = self._train_data
            assert len(x1) == len(y1)

            temp = np.random.permutation(len(y))
            temp1 = np.random.permutation(len(y1))

            self._test_data = x[temp], y[temp]
            self._train_data = x1[temp1], y1[temp1]

    def next_batch_from_test(self, batch_size=-1):
        """returns the next batch to test
        
        Keyword Arguments:
            batch_size {int} -- size of batch (default: {-1})
        
        Raises:
            ValueError -- missing batch size
            RuntimeError -- epoch exceeds 1
        
        Returns:
            np.array -- the testing batch
        """

        if batch_size == -1 and self.batch_size == 0:
            raise ValueError('You must specify the batch size')
        else:
            if batch_size == -1:
                batch_size = self.batch_size
        (test_data_x, test_data_y) = self._test_data
        assert len(test_data_x) == len(test_data_y)
        if batch_size+self._last_batch_test < len(test_data_x):
            start = self._last_batch_test
            stop = batch_size+start
            self._last_batch_test = stop
            return test_data_x[start:stop], test_data_y[start:stop]
        elif not self._last_test_batch_shown:
            self._epoch_count += 1
            warnings.warn(
                'This should be the last call to next_batch_test. Completed the iteration')
            self._last_test_batch_shown = True
            start = self._last_batch_test
            self._last_batch_test = len(test_data_x)
            return test_data_x[start:], test_data_y[start:]
        else:
            raise RuntimeError('Test data was exhaused and still')
            # It is better to stop evalution iteration as soon as this error is thrown

    def next_batch_from_train(self, batch_size=-1):
        """returns next batch from train set

        Keyword Arguments:
            batch_size {int} -- batch size to return (default: {-1})

        Raises:
            ValueError -- if batch size is unspecified

        Returns:
            np.array -- the dataset numpy array
        """

        if batch_size == -1 and self.batch_size == 0:
            raise ValueError('You must specify the batch size')
        else:
            if batch_size == -1:
                batch_size = self.batch_size
        (train_data_x, train_data_y) = self._train_data
        assert len(train_data_x) == len(train_data_y)

        if batch_size+self._last_batch_train < len(train_data_x):
            start = self._last_batch_train
            stop = batch_size+start
            self._last_batch_train = stop
            return train_data_x[start:stop], train_data_y[start:stop]

        else:
            start = self._last_batch_train-len(train_data_x)
            stop = batch_size - (math.fabs(start))
            self._epoch_count += 1
            return train_data_x[start:stop], train_data_y[start:stop]
