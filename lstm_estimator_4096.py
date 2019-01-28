import tensorflow as tf
from hooks import ModelStepTrackerHook

def build_dropout_lstm_cell(num_units, activation, reuse,
                            dtype, timesteps, dropout_rate,
                            name, mode):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                   activation=activation,
                                   reuse=reuse,
                                   dtype=dtype,
                                   name=name + '_cell')
    #cell = tf.contrib.rnn.AttentionCellWrapper(cell,
    #                                           attn_length=16,
    #                                           reuse=reuse,
    #                                           state_is_tuple=True)
    if mode == tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_rate,
                                             variational_recurrent=True,
                                             input_size=1,
                                             dtype=dtype)
    return cell
        
def lstm_4096_model_fn(features, labels, mode, params):
    
    batch_size = params['batch_size']
    timesteps = params['timesteps']
    n_feats = params['n_feats']
    feature_columns = params['feature_columns']
    lstm_cell_size = params['lstm_cell_size']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    lambda_l2_reg = params['lambda_l2_reg']
    dense_size = params['dense_size']
    time_pool = params['time_pool']
    cnn_size = params['cnn_size']
    
    N_sub_batches = int(4096 / timesteps)
    
    if not all(map(lambda x: x == lstm_cell_size[-1], lstm_cell_size)):
        raise Exception('All LSTM cells have to be the same size in the CudnnLSTM, got: ' + str(lstm_cell_size))
    
    print('FEATURES')
    print(features)
    print('-' * 20 + '\nLABELS')
    print(labels)
    print('-' * 20 + '\nMODE')
    print(mode)
    print('-' * 20 + '\nPARAMS')
    print(params)
    print('-' * 20)
    
    # Create input layer from features and feature columns
    with tf.variable_scope('input'):
        input_layer = tf.reshape(tf.feature_column.input_layer(features, feature_columns), (batch_size, timesteps)) #(batch_size, timesteps)
    
    with tf.variable_scope('rnn_input'):
        # take max absolute val every time_pool steps
        # for 4096 input w/ time_pool=8, we take abs max of every 8 timesteps for a resulting dimension of (batch_size, 4096/8=512)
        num_splits = int(timesteps/time_pool)
        rnn_input = tf.cast(input_layer, tf.float32)
        print('num_splits=%d (this is the number of timesteps fed to RNN)' % num_splits)
        rnn_input = tf.split(input_layer, num_or_size_splits=num_splits, axis=1)               #([batch_size, time_pool] * num_splits)
        rnn_input = tf.stack(rnn_input, axis=1)                                                #(batch_size, num_splits, time_pool))
        am = tf.argmax(tf.abs(rnn_input), axis=2)                                              #(batch_size, num_splits)
        am_expanded = tf.one_hot(am, depth=time_pool, axis=-1)                                 #(batch_size, num_splits, time_pool)
        rnn_input = tf.reduce_sum(tf.multiply(am_expanded, rnn_input), axis=2, keepdims=True)  #(batch_size, num_splits)
        print('rnn_input', rnn_input)
    
    with tf.variable_scope('cnn_input'):
        # just reshape for CNN
        cnn_input = tf.cast(tf.reshape(input_layer, (batch_size, timesteps, 1)), tf.float32)  #(batch_size, timesteps, 1)
        print('cnn_input', cnn_input)
    
    # Create LSTM layers
    with tf.variable_scope('lstm_structures', reuse=tf.AUTO_REUSE):
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(lstm_cell_size),
                                              lstm_cell_size[-1],
                                              dropout=dropout_rate,
                                              direction='bidirectional',
                                              dtype=tf.float32)
        
    with tf.variable_scope('step_counter', reuse=tf.AUTO_REUSE):
        step_counter = tf.get_variable('step_counter', dtype=tf.int64, shape=[])

    
    with tf.variable_scope('bi_rnn'):
        outputs, state = lstm(rnn_input)
    
    print('rnn_outputs', outputs)
    final_rnn_output = tf.reshape(tf.concat(outputs, axis=2), (batch_size, 2 * num_splits * lstm_cell_size[-1]))
    
    # Create CNN layers
    with tf.variable_scope('cnn_structures'):
        cnn = cnn_input
        for cnn_filter, cnn_kernel in cnn_size:
            cnn = tf.layers.Conv1D(filters=cnn_filter, kernel_size=cnn_kernel, padding='same', activation=tf.nn.leaky_relu)(cnn)
            cnn = tf.layers.MaxPooling1D(2, 2)(cnn)
            if mode == tf.estimator.ModeKeys.TRAIN:
                cnn = tf.layers.Dropout(rate=dropout_rate)(cnn)
            
    print(cnn)
    final_cnn_output = tf.reshape(cnn, (batch_size, int(timesteps) * int(int(cnn_size[-1][0]) / int(2**len(cnn_size)))))
    print(final_cnn_output)
    
    dense = tf.cast(tf.concat([final_rnn_output, final_cnn_output], axis=1), tf.float64)
    
    # Pass through a Dense layer(s)
    with tf.variable_scope('dense_head'):
        for i, dense_sz in enumerate(dense_size):
            dense = tf.layers.Dense(units=dense_sz,
                                    activation=tf.nn.leaky_relu,
                                    name='dense_layer_%d' % i,
                                    dtype=tf.float64)(dense)

            if mode == tf.estimator.ModeKeys.TRAIN:
                dense = tf.layers.Dropout(rate=dropout_rate)(dense)
            
        dense = tf.layers.Dense(units=timesteps,
                                activation=None,
                                name='dense_layer_out',
                                dtype=tf.float64)(dense)
    
    # Reshape dense outputs to same shape as `labels'
    results = tf.reshape(dense, (batch_size, timesteps))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': results
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, prediction_hooks=[ModelStepTrackerHook()])
        
    # Compute loss.
    with tf.variable_scope('loss_sse'):
        loss = tf.reduce_sum(tf.square(labels - results), name='loss_sse')
        loss_spacing = 0.005 * tf.reduce_sum(tf.abs(labels[:,1:] - labels[:,:-1]) - tf.constant(1.1e-9, dtype=tf.float64))
        loss += loss_spacing
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf.cast(tf_var, tf.float64))
                for tf_var in tf.trainable_variables()
                if not ("noreg" in tf_var.name or "Bias" in tf_var.name \
                        or "input" in tf_var.name or "step_counter" in tf_var.name \
                        or "bias" in tf_var.name)
        )
        loss += l2

    # Compute evaluation metrics.
    mse_op = tf.metrics.mean_squared_error(labels=labels[:,-1],
                                           predictions=results[:,-1],
                                           name='mse_op')
    mae_op = tf.metrics.mean_absolute_error(labels=labels[:,-1],
                                            predictions=results[:,-1],
                                            name='mae_op')
    tf.summary.scalar('mse', mse_op[1])
    tf.summary.scalar('mae', mae_op[1])
    metrics = {'mse': mse_op, 'mae': mae_op}
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[ModelStepTrackerHook()])
    
    
    assert mode == tf.estimator.ModeKeys.TRAIN, 'invalid mode key: ' + str(mode)
    
    grad_dtype_map = {}
    
    # Create train op
    with tf.variable_scope('optimization'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        gradient_var_pairs = optimizer.compute_gradients(loss)
        gvps = list(filter(lambda gvp: gvp[0] is not None, gradient_var_pairs))
        vars = [x[1] for x in gvps]
        grad_dtypes = [x[0].dtype for x in gvps]
        gradients = [tf.cast(x[0], tf.float64) for x in gvps]
        #print('GRADIENTS:\n\t' + '\n\t'.join(list(map(str, gradients))))
        clipped, _ = tf.clip_by_global_norm(gradients, 0.5)
        #print('CLIPPED  :\n\t' + '\n\t'.join(list(map(str, clipped))))
        tf.summary.histogram('grad_norm', tf.global_norm(clipped))
        clipped = [tf.cast(g, grad_dtypes[i]) for i, g in enumerate(clipped)]
        train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=tf.train.get_global_step())
        #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[ModelStepTrackerHook()])
