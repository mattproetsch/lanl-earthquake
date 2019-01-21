import tensorflow as tf

def lstm_model_fn(features, labels, mode, params):
    
    batch_size = params['batch_size']
    timesteps = params['timesteps']
    n_feats = params['n_feats']
    feature_columns = params['feature_columns']
    lstm_cell_size = params['lstm_cell_size']
    learning_rate = params['learning_rate']
    #n_fwd = params['num_fwd_lstm_cells']
    #n_bck = params['num_bck_lstm_cells']
    
    # Create input layer from features and feature columns
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    
    # Create LSTM layers
    lstm_cell_fwd = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size,
                                            activation=tf.nn.leaky_relu,
                                            reuse=False,
                                            dtype=tf.float64,
                                            name='lstm_cell')
    
    lstm_cell_bck = tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size,
                                            activation=tf.nn.leaky_relu,
                                            reuse=False,
                                            dtype=tf.float64,
                                            name='lstm_cell')
    
    init_state = [lstm_cell_fwd.zero_state(batch_size, dtype=tf.float64),
                  lstm_cell_bck.zero_state(batch_size, dtype=tf.float64)]
    
    # Set up variable schemes
    with tf.variable_scope('lstm_state', reuse=tf.AUTO_REUSE):
        init_state_fwd_c_zero = tf.get_variable('init_state_fwd_c_zero', 
                                                shape=(batch_size, lstm_cell_size),
                                                dtype=tf.float64,
                                                trainable=False)
        init_state_fwd_h_zero = tf.get_variable('init_state_fwd_h_zero', 
                                                shape=(batch_size, lstm_cell_size),
                                                dtype=tf.float64,
                                                trainable=False)
        init_state_bck_c_zero = tf.get_variable('init_state_bck_c_zero', 
                                                shape=(batch_size, lstm_cell_size),
                                                dtype=tf.float64,
                                                trainable=False)
        init_state_bck_h_zero = tf.get_variable('init_state_bck_h_zero', 
                                                shape=(batch_size, lstm_cell_size),
                                                dtype=tf.float64,
                                                trainable=False)
        
        init_state_fwd_c = tf.get_variable('init_state_fwd_c',
                                           shape=(batch_size, lstm_cell_size),
                                           dtype=tf.float64,
                                           trainable=False)
        init_state_fwd_h = tf.get_variable('init_state_fwd_h',
                                           shape=(batch_size, lstm_cell_size),
                                           dtype=tf.float64,
                                           trainable=False)
        init_state_bck_c = tf.get_variable('init_state_bck_c',
                                           shape=(batch_size, lstm_cell_size),
                                           dtype=tf.float64,
                                           trainable=False)
        init_state_bck_h = tf.get_variable('init_state_bck_h',
                                           shape=(batch_size, lstm_cell_size),
                                           dtype=tf.float64,
                                           trainable=False)
        
        tf.assign(init_state_fwd_c_zero, init_state[0][0])
        tf.assign(init_state_fwd_h_zero, init_state[0][1])
        tf.assign(init_state_bck_c_zero, init_state[1][0])
        tf.assign(init_state_bck_h_zero, init_state[1][1])
        
        tf.assign(init_state_fwd_c, init_state_fwd_c_zero)
        tf.assign(init_state_fwd_h, init_state_fwd_h_zero)
        tf.assign(init_state_bck_c, init_state_bck_c_zero)
        tf.assign(init_state_bck_h, init_state_bck_h_zero)
        
        
    
    N_sub_batches = int(4096 / timesteps)
    assign_fwd_c = tf.assign(init_state_fwd_c, tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                             true_fn=lambda: init_state_fwd_c_zero,
                             false_fn=lambda: init_state_fwd_c))
    assign_fwd_h = tf.assign(init_state_fwd_h, tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                             true_fn=lambda: init_state_fwd_h_zero,
                             false_fn=lambda: init_state_fwd_h))
    assign_bck_c = tf.assign(init_state_bck_c, tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                             true_fn=lambda: init_state_bck_c_zero,
                             false_fn=lambda: init_state_bck_c))
    assign_bck_h = tf.assign(init_state_bck_h, tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                             true_fn=lambda: init_state_bck_h_zero,
                             false_fn=lambda: init_state_bck_h))
    
    # Prepare sub-batching for RNN state propagation
    rnn_input = tf.cast(tf.reshape(input_layer, (batch_size, timesteps, 1)), tf.float64)
    
    with tf.control_dependencies([assign_fwd_c, assign_fwd_h, assign_bck_c, assign_bck_h]):
        #rnn_inputs_by_timestep = [rnn_inputs[:,i,:n_feats] for i in range(timesteps)]
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fwd,
                                                            cell_bw=lstm_cell_bck,
                                                            initial_state_fw=tf.nn.rnn_cell.LSTMStateTuple(init_state_fwd_c,
                                                                                                           init_state_fwd_h),
                                                            initial_state_bw=tf.nn.rnn_cell.LSTMStateTuple(init_state_bck_c,
                                                                                                           init_state_bck_h),
                                                            inputs=rnn_input,
                                                            dtype=tf.float64)
        tf.assign(init_state_fwd_c, states[0][0])
        tf.assign(init_state_fwd_h, states[0][1])
        tf.assign(init_state_bck_c, states[1][0])
        tf.assign(init_state_bck_h, states[1][1])
                             
    final_rnn_output = tf.reshape(tf.concat(outputs, axis=0), (batch_size, 2 * timesteps * lstm_cell_size))
        
        
    
    # Pass through a Dense layer
    labels_shape = tf.shape(labels)
    dense = tf.layers.Dense(units=timesteps,
                            activation=None,
                            name='dense_layer',
                            dtype=tf.float64)(final_rnn_output)
    
    # Reshape dense outputs to same shape as `labels'
    results = tf.reshape(dense, tf.shape(labels))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': results,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    # Compute loss.
    loss = tf.reduce_sum(tf.square(labels - results), name='loss_sse')

    # Compute evaluation metrics.
    mse_op = tf.metrics.mean_squared_error(labels=labels,
                                           predictions=results,
                                           name='mse_op')
    metrics = {'mse': mse_op}
    tf.summary.scalar('mse', mse_op[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=[metrics])
    
    
    assert mode == tf.estimator.ModeKeys.TRAIN, 'invalid mode key: ' + str(mode)
    # Create train op
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

