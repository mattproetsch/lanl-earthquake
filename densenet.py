import tensorflow as tf
import numpy as np
from hooks import SlimVarAnalyzer
from process_inputs import to_timepool
import re


class DenseNet(object):
    """
    Implementation of "Densely Connected Convolutional Network"
    https://arxiv.org/pdf/1608.06993.pdf
    reference implementation:
    https://github.com/taki0112/Densenet-Tensorflow/blob/dd2db93c529455beb36edf5c40bc5e236b7b1a79/MNIST/Densenet_MNIST.py
    """

    def __init__(self, batch_size, timesteps, growth_rate, layer_sizes, training, dropout_rate,
                 compression_theta, dense_size):
        self.batch_size = batch_size
        self.timesteps = timesteps
        self.growth_rate = growth_rate
        self.layer_sizes = layer_sizes
        self.training = training
        self.dropout_rate = dropout_rate
        self.theta = compression_theta
        self.dense_size = dense_size


    def __call__(self, inputs):
        return self._dense_net(inputs)


    def _avg_pool_1d(self, tens, pool_size, strides, name, dtype=tf.float64, padding='VALID'):
        return tf.layers.AveragePooling1D(pool_size=pool_size,
                                          strides=strides,
                                          name=name,
                                          dtype=dtype,
                                          padding=padding)(tens)


    def _global_avg_pool_1d(self, tens):
        return tf.keras.layers.GlobalAveragePooling1D()(tens)


    def _conv_1d(self, tens, filters, kernel_size, strides, name, dtype=tf.float64, padding='SAME'):
        return tf.layers.Conv1D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                dtype=dtype,
                                name=name)(tens)


    def _dense(self, tens, units, name):
        return tf.layers.Dense(units=units, name=name)(tens)


    def _batch_norm(self, tens, name):
        return tf.layers.batch_normalization(tens, name=name, training=self.training)


    def _relu(self, tens):
        return tf.nn.leaky_relu(tens)


    def _dropout(self, tens, rate):
        return tf.layers.Dropout(rate=rate)(tens)


    def _input_layer(self, x):
        with tf.variable_scope('input_layer'):
            x = self._conv_1d(x, self.growth_rate * 2, 7, 2, 'conv1d')
            x = self._avg_pool_1d(x, 3, 2, 'avg_pool')
        return x


    def _bottleneck_layer(self, x, bottleneck_layer_id):
        with tf.variable_scope('bottleneck_' + str(bottleneck_layer_id)):
            x = self._batch_norm(x, '0_batchnorm')
            x = self._relu(x)
            x = self._conv_1d(x, 4 * self.growth_rate, 1, 1, '0_conv1d')
            x = self._dropout(x, self.dropout_rate)
            
            x = self._batch_norm(x, '1_batchnorm')
            x = self._relu(x)
            x = self._conv_1d(x, self.growth_rate, 3, 1, '1_conv1d')
            x = self._dropout(x, self.dropout_rate)
        return x


    def _transition_layer(self, x, transition_layer_id):
        with tf.variable_scope('transition_' + str(transition_layer_id)):
            x = self._batch_norm(x, 'batchnorm')
            x = self._relu(x)
            out_channels = int(x.get_shape().as_list()[-1] * self.theta)
            x = self._conv_1d(x, out_channels, 1, 1, 'conv1d')
            x = self._dropout(x, self.dropout_rate)
            x = self._avg_pool_1d(x, 2, 2, 'avg_pool')
        return x


    def _dense_block(self, x, num_layers, dense_block_id):
        with tf.variable_scope('dense_block_' + str(dense_block_id)):
            layer_outputs = [x]
            for i in range(num_layers):
                next_layer_input = tf.concat(layer_outputs, axis=2)
                next_layer_output = self._bottleneck_layer(next_layer_input, i)
                layer_outputs.append(next_layer_output)
        return tf.concat(layer_outputs, axis=2)


    def _dense_net(self, x):
        x = self._input_layer(x)
        for block_id, layer_size in enumerate(self.layer_sizes[:-1]):
            x = self._dense_block(x, layer_size, block_id)
            x = self._transition_layer(x, block_id)
            
        # no transition layer after final block, so handle it outside the loop
        x = self._dense_block(x, self.layer_sizes[-1], len(self.layer_sizes) - 1)
        
        # output preparation and processing
        x = self._batch_norm(x, 'pre_output_batchnorm')
        x = self._relu(x)
        x = self._global_avg_pool_1d(x)
        x = self._dense(x, self.dense_size, 'pre_output_dense')
        x = self._batch_norm(x, 'output_batchnorm')
        x = self._relu(x)
        x = self._dense(x, 1, 'output_dense')
        return x


def clean_varname(varname):
    varname = re.sub(r'[^A-Za-z0-9]', '_', varname)
    varname = re.sub(r'^_', '', varname)
    varname = re.sub(r'_$', '', varname)
    varname = re.sub(r'__+', '_', varname)
    return varname


def densenet_model_fn(features, labels, mode, params):
    batch_size = params['batch_size']
    timesteps = params['timesteps']
    time_pool = params['time_pool']
    growth_rate = params['growth_rate']
    layer_sizes = params['layer_sizes']
    dropout_rate = params['dropout_rate']
    theta = params['compression_theta']
    dense_size = params['dense_size']
    feature_columns = params['feature_columns']
    optimizer_name = params['optimizer_name']
    learning_rate = params['learning_rate']
    lambda_l2_reg = params['lambda_l2_reg']
    grad_clip = params['grad_clip']

    training = mode == tf.estimator.ModeKeys.TRAIN


    # extract input
    input_column_features = tf.feature_column.input_layer(features, feature_columns)
    with tf.variable_scope('input_ops'):
        input_column_features = to_timepool(input_column_features, timesteps, time_pool)
    # create and run network
    with tf.device('/gpu:0'):
        dense_net = DenseNet(batch_size, timesteps, growth_rate, layer_sizes,
                         training, dropout_rate, theta, dense_size)
        results = dense_net(input_column_features)

    target = labels[:,-1]
    eval_metrics = {}
    print('results', results)
    print('labels', labels)
    print('target', target)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': results,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, prediction_hooks=[SlimVarAnalyzer()])


    with tf.variable_scope('loss_sse'):
        # calculate sum of squared errors
        loss = tf.reduce_sum(tf.square(target - results), name='loss_sse')
        # loss += tf.reduce_sum(tf.square(input_feats - results), name='loss_ae')
        l2_penalty = tf.cast(tf.train.exponential_decay(lambda_l2_reg,
                                                        tf.train.get_global_step(),
                                                        400,
                                                        0.92,
                                                        staircase=True), tf.float64)
        l2 = l2_penalty * sum(
            tf.nn.l2_loss(tf.cast(tf_var, tf.float64))
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name \
                        or "input" in tf_var.name or "step_counter" in tf_var.name \
                        or "bias" in tf_var.name)
        )
        loss += l2
        tf.summary.scalar('l2_penalty', l2_penalty)
        tf.summary.scalar('l2', l2)

    # Compute evaluation metrics.
    mse_op = tf.metrics.mean_squared_error(labels=target, #labels=input_feats[:,-1]
                                           predictions=results,
                                           name='mse_op')
    mae_op = tf.metrics.mean_absolute_error(labels=target, #labels=input_feats[:,-1]
                                            predictions=results,
                                            name='mae_op')
    tf.summary.scalar('mse', mse_op[1])
    tf.summary.scalar('mae', mae_op[1])
    eval_metrics.update({'mse': mse_op, 'mae': mae_op})

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)


    assert mode == tf.estimator.ModeKeys.TRAIN, 'invalid mode key: ' + str(mode)

    grad_dtype_map = {}

    # Create train op
    with tf.variable_scope('optimization'):
        learning_rate = tf.train.exponential_decay(learning_rate,
                                                   tf.train.get_global_step(),
                                                   400,
                                                   0.96,
                                                   staircase=True)
        tf.summary.scalar('loss_sse/learning_rate', learning_rate)
        if optimizer_name == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == 'RMSProp':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            raise Exception('Unsupported optimizer: ' + str(optimizer))
        #return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimizer.minimize(loss, global_step=tf.train.get_global_step()))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradient_var_pairs = optimizer.compute_gradients(loss)
            #gvps = gradient_var_pairs 
            gvps = list(filter(lambda gvp: gvp[0] is not None, gradient_var_pairs))
            tfvars = [x[1] for x in gvps]
            grad_dtypes = [x[0].dtype for x in gvps]
            gradients = [tf.cast(x[0], tf.float64) for x in gvps]
            #print('GRADIENTS:\n\t' + '\n\t'.join(list(map(str, gradients))))
            clipped, _ = tf.clip_by_global_norm(gradients, grad_clip)
            #print('CLIPPED  :\n\t' + '\n\t'.join(list(map(str, clipped))))
            tf.summary.histogram('grad_norm', tf.global_norm(clipped))
            for x, y in zip(clipped, tfvars):
                tf.summary.histogram('VAR_' + clean_varname(str(y)), y)
                tf.summary.histogram('GRAD_' + clean_varname(str(y)), x)
            clipped = [tf.cast(g, grad_dtypes[i]) for i, g in enumerate(clipped)]
            train_op = optimizer.apply_gradients(zip(clipped, tfvars), global_step=tf.train.get_global_step())
            #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[SlimVarAnalyzer()])
    