import tensorflow as tf

def lstm_model_fn(features, labels, mode, params):
    
    batch_size = params['batch_size']
    timesteps = params['timesteps']
    n_feats = params['n_feats']
    feature_columns = params['feature_columns']
    lstm_cell_size = params['lstm_cell_size']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    
    print('FEATURES')
    print(features)
    print('LABELS')
    print(labels)
    print('MODE')
    print(mode)
    print('PARAMS')
    print(params)
    
    # Create input layer from features and feature columns
    input_layer = tf.feature_column.input_layer(features, feature_columns)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        labels_input_layer = tf.feature_column.input_layer(features, params['label_input_column'], trainable=False)
        labels_input_layer = tf.reshape(labels_input_layer, (batch_size, timesteps))
    
    # Create LSTM layers
    lstm_cell_fwd_cells = [tf.nn.rnn_cell.LSTMCell(num_units=cell_sz,
                                            activation=tf.nn.leaky_relu,
                                            reuse=False,
                                            dtype=tf.float64,
                                            name='lstm_cell') for cell_sz in lstm_cell_size]
    
    lstm_cell_bck_cells = [tf.nn.rnn_cell.LSTMCell(num_units=cell_sz,
                                            activation=tf.nn.leaky_relu,
                                            reuse=False,
                                            dtype=tf.float64,
                                            name='lstm_cell') for cell_sz in lstm_cell_size]
    
    lstm_cell_fwd_m = tf.nn.rnn_cell.MultiRNNCell(lstm_cell_fwd_cells)
    lstm_cell_bck_m = tf.nn.rnn_cell.MultiRNNCell(lstm_cell_bck_cells)
    
    init_state = [lstm_cell_fwd_m.zero_state(batch_size, dtype=tf.float64),
                  lstm_cell_bck_m.zero_state(batch_size, dtype=tf.float64)]
    
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        lstm_cell_fwd = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fwd_m,
                                                      input_keep_prob=dropout_rate,
                                                      variational_recurrent=True,
                                                      input_size=1,
                                                      dtype=tf.float64)
        lstm_cell_bck = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bck_m,
                                                      input_keep_prob=dropout_rate,
                                                      variational_recurrent=True,
                                                      input_size=1,
                                                      dtype=tf.float64)
    
    # Set up variable schemes
    with tf.variable_scope('lstm_state', reuse=tf.AUTO_REUSE):
        init_state_fwd_c_zeros = [tf.get_variable('init_state_fwd_c_zero_%d' % i, 
                                                  shape=(batch_size, cell_sz),
                                                  dtype=tf.float64,
                                                  trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_fwd_h_zeros = [tf.get_variable('init_state_fwd_h_zero_%d' % i, 
                                                  shape=(batch_size, cell_sz),
                                                  dtype=tf.float64,
                                                  trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_bck_c_zeros = [tf.get_variable('init_state_bck_c_zero_%d' % i, 
                                                  shape=(batch_size, cell_sz),
                                                  dtype=tf.float64,
                                                  trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_bck_h_zeros = [tf.get_variable('init_state_bck_h_zero_%d' % i, 
                                                  shape=(batch_size, cell_sz),
                                                  dtype=tf.float64,
                                                  trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        for i in range(len(lstm_cell_size)):
            tf.assign(init_state_fwd_c_zeros[i], init_state[0][i][0])
            tf.assign(init_state_fwd_h_zeros[i], init_state[0][i][1])
            tf.assign(init_state_bck_c_zeros[i], init_state[1][i][0])
            tf.assign(init_state_bck_h_zeros[i], init_state[1][i][1])
        
        init_state_fwd_cs = [tf.get_variable('init_state_fwd_c_%d' % i, 
                                             shape=(batch_size, cell_sz),
                                             dtype=tf.float64,
                                             trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_fwd_hs = [tf.get_variable('init_state_fwd_h_%d' % i, 
                                             shape=(batch_size, cell_sz),
                                             dtype=tf.float64,
                                             trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_bck_cs = [tf.get_variable('init_state_bck_c_%d' % i, 
                                             shape=(batch_size, cell_sz),
                                             dtype=tf.float64,
                                             trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        init_state_bck_hs = [tf.get_variable('init_state_bck_h_%d' % i, 
                                             shape=(batch_size, cell_sz),
                                             dtype=tf.float64,
                                             trainable=False) for i, cell_sz in enumerate(lstm_cell_size)]
        
        for i in range(len(lstm_cell_size)):
            tf.assign(init_state_fwd_cs[i], init_state_fwd_c_zeros[i])
            tf.assign(init_state_fwd_hs[i], init_state_fwd_h_zeros[i])
            tf.assign(init_state_bck_cs[i], init_state_bck_c_zeros[i])
            tf.assign(init_state_bck_hs[i], init_state_bck_h_zeros[i])
        
        
    
    N_sub_batches = int(4096 / timesteps)
    
    control_ops = []
    for i in range(len(lstm_cell_size)):
        assign_fwd_c = tf.assign(init_state_fwd_cs[i], tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                                                               true_fn=lambda: init_state_fwd_c_zeros[i],
                                                               false_fn=lambda: init_state_fwd_cs[i]))
        assign_fwd_h = tf.assign(init_state_fwd_hs[i], tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                                                               true_fn=lambda: init_state_fwd_h_zeros[i],
                                                               false_fn=lambda: init_state_fwd_hs[i]))
        assign_bck_c = tf.assign(init_state_bck_cs[i], tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                                                               true_fn=lambda: init_state_bck_c_zeros[i],
                                                               false_fn=lambda: init_state_bck_cs[i]))
        assign_bck_h = tf.assign(init_state_bck_hs[i], tf.cond(tf.equal(tf.floormod(tf.train.get_global_step(), N_sub_batches), 0),
                                                               true_fn=lambda: init_state_bck_h_zeros[i],
                                                               false_fn=lambda: init_state_bck_hs[i]))
        
        control_ops.extend([assign_fwd_c, assign_fwd_h, assign_bck_c, assign_bck_h])
    
    print(control_ops)
    
    # Prepare sub-batching for RNN state propagation
    rnn_input = tf.cast(tf.reshape(input_layer, (batch_size, timesteps, 1)), tf.float64)
    
    with tf.control_dependencies(control_ops):
        #rnn_inputs_by_timestep = [rnn_inputs[:,i,:n_feats] for i in range(timesteps)]
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fwd_m,
                                                            cell_bw=lstm_cell_bck_m,
                                                            initial_state_fw=tuple([tf.nn.rnn_cell.LSTMStateTuple(init_state_fwd_cs[i],
                                                                                                                  init_state_fwd_hs[i])
                                                                                    for i in range(len(lstm_cell_size))]),
                                                            initial_state_bw=tuple([tf.nn.rnn_cell.LSTMStateTuple(init_state_bck_cs[i],
                                                                                                                  init_state_bck_hs[i])
                                                                                    for i in range(len(lstm_cell_size))]),
                                                            inputs=rnn_input,
                                                            dtype=tf.float64)
        
        for i in range(len(lstm_cell_size)):
            tf.assign(init_state_fwd_cs[i], states[0][i][0])
            tf.assign(init_state_fwd_hs[i], states[0][i][1])
            tf.assign(init_state_bck_cs[i], states[1][i][0])
            tf.assign(init_state_bck_hs[i], states[1][i][1])
    
    final_rnn_output = tf.reshape(tf.concat(outputs, axis=0), (batch_size, 2 * timesteps * lstm_cell_size[-1]))
    
    
    # Pass through a Dense layer
    dense = tf.layers.Dense(units=timesteps,
                            activation=None,
                            name='dense_layer',
                            dtype=tf.float64)(final_rnn_output)
    
    # Reshape dense outputs to same shape as `labels'
    results = tf.reshape(dense, (batch_size, timesteps))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'labels': labels_input_layer,
            'pred': results
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    # Compute loss.
    loss = tf.reduce_sum(tf.square(labels - results), name='loss_sse')

    # Compute evaluation metrics.
    mse_op = tf.metrics.mean_squared_error(labels=labels,
                                           predictions=results,
                                           name='mse_op')
    mae_op = tf.metrics.mean_absolute_error(labels=labels,
                                            predictions=results,
                                            name='mae_op')
    tf.summary.scalar('mse', mse_op[1])
    tf.summary.scalar('mae', mae_op[1])
    metrics = {'mse': mse_op, 'mae': mae_op}
    
    

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    
    
    assert mode == tf.estimator.ModeKeys.TRAIN, 'invalid mode key: ' + str(mode)
    # Create train op
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

