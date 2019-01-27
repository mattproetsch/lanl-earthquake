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
    #                                           attn_length=timesteps,
    #                                           reuse=reuse,
    #                                           state_is_tuple=True)    
    if mode == tf.estimator.ModeKeys.TRAIN:
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_rate,
                                             variational_recurrent=True,
                                             input_size=1,
                                             dtype=tf.float64)
    return cell
        
def lstm_model_fn(features, labels, mode, params):
    
    batch_size = params['batch_size']
    timesteps = params['timesteps']
    n_feats = params['n_feats']
    feature_columns = params['feature_columns']
    lstm_cell_size = params['lstm_cell_size']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    lambda_l2_reg = params['lambda_l2_reg']
    
    N_sub_batches = int(4096 / timesteps)
    
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
        input_layer = tf.feature_column.input_layer(features, feature_columns)
        rnn_input = tf.cast(tf.reshape(input_layer, (batch_size, timesteps, 1)), tf.float64)
    
    # Create LSTM layers
    lstm_cells_fwd = []
    lstm_cells_bck = []
    with tf.variable_scope('lstm_structures', reuse=tf.AUTO_REUSE):
        for i, cell_sz in enumerate(lstm_cell_size):
            
            new_cell_fwd = build_dropout_lstm_cell(cell_sz, activation=tf.nn.leaky_relu, reuse=False,
                                                   dtype=tf.float64, timesteps=timesteps, dropout_rate=dropout_rate,
                                                   name='lstm_cell_fwd_%d' % i, mode=mode)
            lstm_cells_fwd.append(new_cell_fwd)
            
            new_cell_bck = build_dropout_lstm_cell(cell_sz, activation=tf.nn.leaky_relu, reuse=False,
                                                   dtype=tf.float64, timesteps=timesteps, dropout_rate=dropout_rate,
                                                   name='lstm_cell_bck_%d' % i, mode=mode)
            lstm_cells_bck.append(new_cell_bck)

        lstm_cell_fwd_m = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_fwd)
        lstm_cell_bck_m = tf.nn.rnn_cell.MultiRNNCell(lstm_cells_bck)

        init_state = [lstm_cell_fwd_m.zero_state(batch_size, dtype=tf.float64),
                      lstm_cell_bck_m.zero_state(batch_size, dtype=tf.float64)]
    
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
        
    with tf.variable_scope('step_counter', reuse=tf.AUTO_REUSE):
        step_counter = tf.get_variable('step_counter', dtype=tf.int64, shape=[])
    
    # Keep track of all the assignment control ops we are about to create
    control_ops = []
    with tf.variable_scope('assign_zeros_at_batch_boundary', reuse=tf.AUTO_REUSE):
        for i in range(len(lstm_cell_size)):
            # Conditionally set the LSTM states to zero states if we are starting a new batch.
            # We start a new batch at every multiple of N_sub_batches, or when floormod(steps, N_sub_batches) == 0.
            assign_fwd_c = tf.assign(init_state_fwd_cs[i], tf.cond(tf.equal(tf.floormod(step_counter, N_sub_batches), 0),
                                                                   true_fn=lambda: init_state_fwd_c_zeros[i],
                                                                   false_fn=lambda: init_state_fwd_cs[i]))
            assign_fwd_h = tf.assign(init_state_fwd_hs[i], tf.cond(tf.equal(tf.floormod(step_counter, N_sub_batches), 0),
                                                                   true_fn=lambda: init_state_fwd_h_zeros[i],
                                                                   false_fn=lambda: init_state_fwd_hs[i]))
            assign_bck_c = tf.assign(init_state_bck_cs[i], tf.cond(tf.equal(tf.floormod(step_counter, N_sub_batches), 0),
                                                                   true_fn=lambda: init_state_bck_c_zeros[i],
                                                                   false_fn=lambda: init_state_bck_cs[i]))
            assign_bck_h = tf.assign(init_state_bck_hs[i], tf.cond(tf.equal(tf.floormod(step_counter, N_sub_batches), 0),
                                                                   true_fn=lambda: init_state_bck_h_zeros[i],
                                                                   false_fn=lambda: init_state_bck_hs[i]))

            control_ops.extend([assign_fwd_c, assign_fwd_h, assign_bck_c, assign_bck_h])
    
    with tf.control_dependencies(control_ops):
        with tf.variable_scope('bi_rnn'):
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
        with tf.variable_scope('assign_states_after_rnn'):
            for i in range(len(lstm_cell_size)):
                tf.assign(init_state_fwd_cs[i], states[0][i][0])
                tf.assign(init_state_fwd_hs[i], states[0][i][1])
                tf.assign(init_state_bck_cs[i], states[1][i][0])
                tf.assign(init_state_bck_hs[i], states[1][i][1])
    
    
    final_rnn_output = tf.reshape(tf.concat(outputs, axis=2), (batch_size, 2 * timesteps * lstm_cell_size[-1]))
    
    # Pass through a Dense layer
    with tf.variable_scope('dense_head'):
        dense = tf.layers.Dense(units=timesteps,
                                activation=tf.nn.leaky_relu,
                                name='dense_layer',
                                dtype=tf.float64)(final_rnn_output)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            dense = tf.layers.Dropout(rate=dropout_rate)(dense)
            
        dense = tf.layers.Dense(units=timesteps,
                                activation=tf.nn.leaky_relu,
                                name='dense_layer2',
                                dtype=tf.float64)(dense)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            dense = tf.layers.Dropout(rate=dropout_rate)(dense)
            
        dense = tf.layers.Dense(units=timesteps,
                                activation=None,
                                name='dense_layer3',
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
        loss = tf.reduce_sum(tf.square(labels[:,-1] - results[:,-1]), name='loss_sse')
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
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
    
    # Create train op
    with tf.variable_scope('optimization'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[ModelStepTrackerHook()])
