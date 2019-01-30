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
    grad_clip = params['grad_clip']
    dense_size = params['dense_size']
    time_pool = params['time_pool']
    cnn_size = params['cnn_size']
    stft_frame_length = params['stft_frame_length']
    stft_frame_step = params['stft_frame_step']
    
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
        with tf.device('/gpu:0'):
            input_layer = tf.reshape(tf.feature_column.input_layer(features, feature_columns), (batch_size, timesteps)) #(batch_size, timesteps)
            
    
    # This code does CNN stuff on the input
    # It didn't work so well
    # with tf.variable_scope('cnn_input'):
    #     # just reshape for CNN
    #     cnn_input = tf.cast(tf.reshape(input_layer, (batch_size, timesteps, 1)), tf.float64)  #(batch_size, timesteps, 1)
    #     print('cnn_input', cnn_input)
    # 
    # # Create CNN layers
    # with tf.variable_scope('cnn_structures'):
    #     cnn = cnn_input
    #     for i, (cnn_filter, cnn_kernel) in enumerate(cnn_size):
    #         cnn = tf.layers.Conv1D(filters=cnn_filter, kernel_size=cnn_kernel,
    #                                padding='same', activation=tf.nn.leaky_relu, dtype=tf.float64)(cnn)
    #         if i <= 3:
    #             cnn = tf.layers.batch_normalization(cnn, training=mode == tf.estimator.ModeKeys.TRAIN)
    #         if i >= 3:
    #             cnn = tf.layers.MaxPooling1D(2, 2)(cnn)
    #         if mode == tf.estimator.ModeKeys.TRAIN:
    #             cnn = tf.layers.Dropout(rate=dropout_rate)(cnn)
    #         
    #         
    # print('cnn (output)', cnn)
    # cnn_output_time = int(int(timesteps) / int(2**(len(cnn_size) - 3)))
    # final_cnn_output = tf.reshape(cnn, (batch_size, int(int(cnn_size[-1][0])) * cnn_output_time))
    # print('reshaped cnn output', final_cnn_output)
    # 
    # # Regularization
    # with tf.variable_scope('cnn_loss'):
    #     cnn_dense = tf.layers.Dense(units=256, activation=tf.nn.leaky_relu, name='cnn_pre_dense', dtype=tf.float64)(final_cnn_output)
    #     cnn_dense_out = tf.layers.Dense(units=1, activation=None, name='cnn_dense_loss_layer', dtype=tf.float64)(cnn_dense)
    #     cnn_unfair_loss = tf.reduce_sum(tf.abs(labels[:,-1] - cnn_dense_out))
    #     cul_op = tf.metrics.mean_absolute_error(labels=labels[:,-1],
    #                                             predictions=cnn_dense_out,
    #                                             name='cul_op')
    #     tf.summary.scalar('unfair_loss_cnn', cul_op[1])
    
    
    
    # This code takes maxval of every time_pool steps to compress the input sequence. it didn't work very well.
    with tf.variable_scope('stride_rnn_input'):
        # take max absolute val every time_pool steps
        # for 4096 input w/ time_pool=8, we take abs max of every 8 timesteps for a resulting dimension of (batch_size, 4096/8=512)
        with tf.device('/gpu:0'):
            num_splits = int(timesteps/time_pool)
            stride_rnn_input = tf.cast(input_layer, tf.float64)
            print('num_splits=%d (this is the number of timesteps fed to RNN)' % num_splits)
            stride_rnn_input = tf.split(stride_rnn_input, num_or_size_splits=num_splits, axis=1)        #([batch_size, time_pool] * num_splits)
            stride_rnn_input = tf.stack(stride_rnn_input, axis=1)                                       #(batch_size, num_splits, time_pool))
            am = tf.argmax(tf.abs(stride_rnn_input), axis=2)                                            #(batch_size, num_splits)
            am_expanded = tf.one_hot(am, depth=time_pool, axis=-1, dtype=tf.float64)                    #(batch_size, num_splits, time_pool)
            stride_rnn_input = tf.reduce_sum(tf.multiply(am_expanded, stride_rnn_input), axis=2, keepdims=True)  #(batch_size, num_splits, 1)
            stride_rnn_input = tf.transpose(stride_rnn_input, [1, 0, 2])                                #(num_splits, batch_size, 1)
            print('stride_rnn_input', stride_rnn_input)
    
    # Create LSTM layers
    with tf.variable_scope('stride_lstm', reuse=tf.AUTO_REUSE):
        with tf.device('/gpu:0'):
            stride_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(lstm_cell_size),
                                                         lstm_cell_size[-1],
                                                         dropout=dropout_rate,
                                                         direction='bidirectional',
                                                         dtype=tf.float64)
            stride_outputs, stride_states = stride_lstm(stride_rnn_input)
        tf.summary.histogram('stride_lstm_states', stride_states)
    
    print('stride_outputs', stride_outputs)
    final_stride_output = tf.transpose(stride_outputs, [1, 0, 2])[:,-1,:]
    tf.summary.histogram('final_stride_output', final_stride_output)
    # Regularization network for stride network
    with tf.variable_scope('stride_rnn_regularization'):
        stride_rnn_dense = tf.layers.Dense(units=256, activation=tf.nn.leaky_relu, name='stride_rnn_dense', dtype=tf.float64)(final_stride_output)
        stride_reg_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(stride_rnn_dense)
        stride_reg_loss = tf.reduce_sum(tf.abs(labels[:,-1] - stride_reg_out))
        stride_mae_op = tf.metrics.mean_absolute_error(labels=labels[:,-1],
                                                       predictions=stride_reg_out,
                                                       name='stride_mae_op')
        tf.summary.scalar('stride_mae', stride_mae_op[1])
    
    
    # more features: STFT
    with tf.variable_scope('stft_rnn_input'):
        stft_rnn_input = tf.reshape(input_layer, [batch_size, timesteps])
        # (batch_size, frame_step, 4096 // frame_length + 1)
        stft_rnn_input = tf.contrib.signal.stft(stft_rnn_input, frame_length=stft_frame_length,
                                                frame_step=stft_frame_step, pad_end='None', name='stft_op')
        stft_rnn_input = tf.real(stft_rnn_input * tf.conj(stft_rnn_input))
        log_offset = 1e-6
        stft_rnn_input = tf.log(stft_rnn_input + log_offset) # take log power, as is tradition
        stft_rnn_input = tf.transpose(stft_rnn_input, [1, 0, 2])
        stft_rnn_input = tf.cast(stft_rnn_input, tf.float64)
        stft_rnn_input = tf.layers.batch_normalization(stft_rnn_input, training=mode == tf.estimator.ModeKeys.TRAIN)
        tf.summary.histogram('stft_rnn_input', stft_rnn_input)
        print('stft_rnn_input', stft_rnn_input)
        
    # Create LSTM layers
    with tf.variable_scope('stft_lstm', reuse=tf.AUTO_REUSE):
        with tf.device('/gpu:0'):
            stft_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(lstm_cell_size),
                                                           lstm_cell_size[-1],
                                                           dropout=dropout_rate,
                                                           direction='bidirectional',
                                                           dtype=tf.float64)
            stft_outputs, stft_states = stft_lstm(stft_rnn_input)
        tf.summary.histogram('stft_lstm_states', stft_states)
    
    print('stft_outputs', stft_outputs)
    #rnn_timesteps = int(4096 / stft_frame_step)
    final_stft_output = tf.transpose(stft_outputs, [1, 0, 2])[:,-1,:]
    tf.summary.histogram('final_stft_output', final_stft_output)
    # Regularization network for stft network
    with tf.variable_scope('stft_rnn_regularization'):
        stft_rnn_dense = tf.layers.Dense(units=256, activation=tf.nn.leaky_relu, name='stft_rnn_dense', dtype=tf.float64)(final_stft_output)
        stft_reg_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(stft_rnn_dense)
        stft_reg_loss = tf.reduce_sum(tf.abs(labels[:,-1] - stft_reg_out))
        stft_mae_op = tf.metrics.mean_absolute_error(labels=labels[:,-1],
                                                     predictions=stft_reg_out,
                                                     name='stft_mae_op')
        tf.summary.scalar('stft_mae', stft_mae_op[1])
    
    
    dense = tf.concat([final_stft_output, final_stride_output], axis=1)
    #dense = tf.cast(final_rnn_output, tf.float64)
    print('dense input', dense)
    
    # Pass through a Dense layer(s)
    
    dense_layers = []
    with tf.variable_scope('dense_head'):
        for i, dense_sz in enumerate(dense_size):
            dense = tf.layers.Dense(units=dense_sz,
                                    activation=tf.nn.leaky_relu,
                                    name='dense_layer_%d' % i,
                                    dtype=tf.float64)(dense)
            
            dense_layers.append(dense)  # Track because we need to return these for prediction (features!)
            
            #if i <= len(dense_size) - 2:
            #    dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)

            if mode == tf.estimator.ModeKeys.TRAIN:
                dense = tf.layers.Dropout(rate=dropout_rate)(dense)
            
        dense = tf.layers.Dense(units=1,
                                activation=None,
                                name='dense_layer_out',
                                dtype=tf.float64)(dense)
    
    # Reshape dense outputs to same shape as `labels'
    results = dense
    print('results', results)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': results,
            'dense_feats': dense_layers[-2]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, prediction_hooks=[ModelStepTrackerHook()])
        
    # Compute loss.
    with tf.variable_scope('loss_sse'):
        # calculate sum of squared errors
        loss = tf.reduce_sum(tf.square(labels[:,-1] - results), name='loss_sse') + stride_reg_loss + stft_reg_loss
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
                                           predictions=results,
                                           name='mse_op')
    mae_op = tf.metrics.mean_absolute_error(labels=labels[:,-1],
                                            predictions=results,
                                            name='mae_op')
    tf.summary.scalar('mse', mse_op[1])
    tf.summary.scalar('mae', mae_op[1])
    metrics = {'mse': mse_op,
               'mae': mae_op,
               'stft_rnn_regularization/stft_mae': stft_mae_op,
               'stride_rnn_regularization/stride_mae': stride_mae_op}
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics, evaluation_hooks=[ModelStepTrackerHook()])
    
    
    assert mode == tf.estimator.ModeKeys.TRAIN, 'invalid mode key: ' + str(mode)
    
    grad_dtype_map = {}
    
    # Create train op
    with tf.variable_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradient_var_pairs = optimizer.compute_gradients(loss)
            gvps = gradient_var_pairs #list(filter(lambda gvp: gvp[0] is not None, gradient_var_pairs))
            vars = [x[1] for x in gvps]
            grad_dtypes = [x[0].dtype for x in gvps]
            gradients = [tf.cast(x[0], tf.float64) for x in gvps]
            #print('GRADIENTS:\n\t' + '\n\t'.join(list(map(str, gradients))))
            clipped, _ = tf.clip_by_global_norm(gradients, grad_clip)
            #print('CLIPPED  :\n\t' + '\n\t'.join(list(map(str, clipped))))
            tf.summary.histogram('grad_norm', tf.global_norm(clipped))
            for x, y in zip(clipped, vars):
                tf.summary.histogram('GRAD_' + str(y), x)
            clipped = [tf.cast(g, grad_dtypes[i]) for i, g in enumerate(clipped)]
            train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=tf.train.get_global_step())
            #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[ModelStepTrackerHook()])
