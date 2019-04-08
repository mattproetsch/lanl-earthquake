import tensorflow as tf
from utils import Loggable

class LSTM(Loggable):
    
    def __init__(self, num_layers, num_units, training, direction='unidirectional',
                       dropout=0.2, dense_units=512, dtype=tf.float64, input_is_batch_first=True,
                       return_only_last_timestep=True, debug=False):
        self.num_layers = num_layers
        self.num_units = num_units
        self.training = training
        self.direction = direction
        self.dropout = dropout
        self.dtype = dtype
        self.input_is_batch_first = input_is_batch_first
        self.return_only_last_timestep = return_only_last_timestep
        self.dense_units = dense_units
        self._DEBUG = debug


    def __call__(self, x):
        x = self._build_lstm(x)
        x = self._output_dense(x)
        return x


    def _lstm(self, x):
        outputs, states = tf.contrib.cudnn_rnn.CudnnLSTM(self.num_layers,
                                                         self.num_units,
                                                         direction=self.direction,
                                                         dropout=self.dropout,
                                                         dtype=self.dtype,
                                                         name='CudnnLSTM')(x)
        return outputs


    def _dense(self, tens, units, name):
        return tf.layers.Dense(units=units, name=name)(tens)


    def _batch_norm(self, tens, name):
        return tens
        return tf.layers.batch_normalization(tens, name=name, training=self.training)


    def _relu(self, tens):
        return tf.nn.leaky_relu(tens)


    def _output_dense(self, x):
        self._log('output_dense', 'x', x)
        with tf.variable_scope('LSTM_dense_output'):
            x = self._dense(x, self.dense_units, 'pre_output_dense')
            self._log('output_dense', 'after_pre_output_dense', x)
            x = self._batch_norm(x, 'pre_output_batchnorm')
            self._log('output_dense', 'after_batchnorm', x)
            x = self._relu(x)
            x = self._dense(x, 1, 'output_dense')
            self._log('output_dense', 'after_output_dense', x)
            x = tf.squeeze(x, axis=-1)
            self._log('output_dense', 'after_squeeze', x)
            return x


    def _build_lstm(self, x):
        self._log('input_layer', 'x', x)
        with tf.variable_scope('LSTM_layer'):
            if self.input_is_batch_first:
                transpose_order = [1, 0, 2]
            else:
                transpose_order = [0, 1, 2]
            x = tf.transpose(x, transpose_order)
            self._log('build_lstm', 'after_transpose', x)
            x = self._lstm(x)
            self._log('build_lstm', 'after_lstm', x)
            x = tf.transpose(x, transpose_order)
            self._log('build_lstm', 'after_2nd_transpose', x)
            if self.return_only_last_timestep:
                x = x[:,-1,:]
            self._log('build_lstm', 'after_slice', x)
        return x