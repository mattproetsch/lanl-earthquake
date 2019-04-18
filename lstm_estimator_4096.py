import tensorflow as tf
from hooks import ModelStepTrackerHook, SlimVarAnalyzer

def build_dropout_lstm_cell(num_units, activation, reuse,
                            dtype, dropout_rate,
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
    dropout_rate = params['dropout_rate'] if mode == tf.estimator.ModeKeys.TRAIN else 0
    lambda_l2_reg = params['lambda_l2_reg']
    grad_clip = params['grad_clip']
    dense_size = params['dense_size']
    time_pool = params['time_pool']
    cnn_spec = params['cnn_spec']
    stft_frame_length = params['stft_frame_length']
    stft_frame_step = params['stft_frame_step']
    use_stft = params['use_stft']
    use_stride = params['use_stride']
    use_stride_cnn = params['use_stride_cnn']
    use_stft_cnn = params['use_stft_cnn']
    regularize_networks = params['regularize_networks']
    dense_batch_norm = params['use_dense_batch_norm']
    use_dense_head = params['use_dense_head']
    use_deconv = params['use_deconv']
    optimizer_name = params['optimizer_name']
    lstm_directionality = params['lstm_directionality']
    directionality_factor = 2 if lstm_directionality == 'bidirectional' else 1
    ae_mode = params['ae_mode']
    
    eval_metrics = {}
    loss = tf.constant(0.0, dtype=tf.float64)
    # _labels = labels[:,-1]
    
    use_attention = False  #TODO #FINDME #FIXME
    
    
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
            input_layer = tf.reshape(tf.feature_column.input_layer(features, feature_columns), (-1, timesteps)) #(batch_size, timesteps)
            
        
    # Compute loss.
    if ae_mode:
        input_feats = tf.cast(input_layer, dtype=tf.float64)
        target = input_feats
    else:
        target = labels
        
    # This code does CNN stuff on the input
    # It didn't work so well
    # (but we're trying again!)
    if use_stride_cnn:
        with tf.variable_scope('stride_cnn_input'):
            num_splits = int(timesteps/time_pool)
            if time_pool == 1:
                stride_cnn_input = tf.reshape(tf.cast(input_layer, tf.float64), (-1, timesteps, 1))
            else:
                stride_cnn_input = tf.cast(to_timepool(input_layer, timesteps, time_pool), tf.float64)
                # stft_cnn_input = tf.cast(to_stft(input_layer, timesteps, stft_frame_length, stft_frame_step), tf.float64)
                # stride_cnn_input = tf.concat([stride_cnn_input, stft_cnn_input], axis=2)
                
            print('stride_cnn_input', stride_cnn_input)
        
        # Create CNN layers
        with tf.variable_scope('stride_cnn_encoder_structures'):
            with tf.device('/gpu:0'):
                trainable = ae_mode  # only update encoder weights when training as autoencoder
                trainable = True  # Uncomment to ALWAYS train
                print('%sllowing trainable encoder' % ('Disa', 'A')[trainable])
                
                stride_cnn = stride_cnn_input
                stride_cnn_activations = [stride_cnn]
                forget_layer = None
                output_layer = None
                if not all(map(lambda x: x['filters'] == cnn_spec[0]['filters'], cnn_spec[:-1])):
                    raise Exception('all CNN filter sizes (except last) should be equal')
                
                for i, cnn_layer_spec in enumerate(cnn_spec[:-1]):
                    s = cnn_layer_spec
                    if s['skip'] is not None:
                        activation = None
                    else:
                        activation = tf.nn.leaky_relu
                    
                    stride_cnn_input = stride_cnn
                    stride = cnn_layer_spec['stride']
                    if i == 0:
                        stride_cnn_activation = tf.layers.Conv1D(filters=s['filters'], kernel_size=s['kernel_size'], strides=stride,
                                                             padding='same', activation=activation, dtype=tf.float64,
                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                                     dtype=tf.float64),
                                                             name='conv-%d' % i, trainable=trainable)(stride_cnn) 
                        print('creating new output/forget gates')
                        forget_layer = tf.layers.Conv1D(filters=cnn_spec[0]['filters'], kernel_size=cnn_spec[0]['kernel_size'], strides=stride,
                                                        padding='same', activation=tf.nn.sigmoid, dtype=tf.float64,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float64),
                                                        name='forget-0', trainable=trainable)
                        output_layer = tf.layers.Conv1D(filters=cnn_spec[0]['filters'], kernel_size=cnn_spec[0]['kernel_size'], strides=stride,
                                                        padding='same', activation=tf.nn.sigmoid, dtype=tf.float64,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float64),
                                                        name='output-0', trainable=trainable)

                        qrnn_c = tf.zeros_like(stride_cnn_activation)
                        
                    else:
                        stride_cnn_activation = tf.layers.Conv1D(filters=s['filters'], kernel_size=s['kernel_size'], strides=stride,
                                                                 padding='same', activation=activation, dtype=tf.float64,
                                                                 kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                                         dtype=tf.float64),
                                                                 name='conv-%d' % i, trainable=trainable)(stride_cnn)
                        if i > 0 and stride_cnn_activations[-1].shape[-1] != stride_cnn_activations[-2].shape[-1] or True:
                            print('creating new output/forget gates')
                            forget_layer = tf.layers.Conv1D(filters=s['filters'], kernel_size=s['kernel_size'], strides=stride,
                                                            padding='same', activation=tf.nn.sigmoid, dtype=tf.float64,
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float64),
                                                            name='forget-%d' % i, trainable=trainable)
                            output_layer = tf.layers.Conv1D(filters=s['filters'], kernel_size=s['kernel_size'], strides=stride,
                                                            padding='same', activation=tf.nn.sigmoid, dtype=tf.float64,
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float64),
                                                            name='output-%d' % i, trainable=trainable)

                    stride_forget = forget_layer(stride_cnn)
                    stride_output = output_layer(stride_cnn)

                    if s['batch_norm']:
                        stride_cnn_activation = tf.layers.batch_normalization(stride_cnn_activation,
                                                                       training=mode == tf.estimator.ModeKeys.TRAIN)

                    if mode == tf.estimator.ModeKeys.TRAIN and s['dropout']:
                        # Apply zoneout-like dropout per Bradbury 2016
                        # fractional dropout
                        stride_forget = 1 - tf.layers.Dropout(rate=((len(cnn_spec) - i) * dropout_rate))(stride_forget)

                    if s['skip'] is not None:
                        # gated skip connections (peephole?)
                        # qrnn_i = stride_input * stride_cnn_activations[-s['skip']] + tf.nn.leaky_relu(stride_cnn_activation)
                        qrnn_i = tf.nn.leaky_relu(stride_cnn_activations[-s['skip']] + stride_cnn_activation)
                    else:
                        qrnn_i = stride_cnn_activation

                    print('forgetting...')
                    qrnn_c = stride_forget * qrnn_c + (1 - stride_forget) * qrnn_i
                    stride_cnn = stride_output * qrnn_c
                    
                    if s['max_pool']:
                        pl = s['max_pool']
                        stride_cnn = tf.layers.MaxPooling1D(pl, pl)(stride_cnn)
                        qrnn_c = tf.layers.MaxPooling1D(pl, pl)(qrnn_c)
                    stride_cnn_activations.append(stride_cnn)
                    print('stride cnn layer %d: %s' % (i, stride_cnn))
                
                # create final layer
                i = len(cnn_spec) - 1
                stride_cnn = tf.layers.Conv1D(filters=cnn_spec[i]['filters'], kernel_size=cnn_spec[i]['kernel_size'],
                                                        padding='same', activation=tf.nn.leaky_relu, dtype=tf.float64,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                                dtype=tf.float64),
                                                        name='conv-%d' % i, trainable=trainable)(stride_cnn)
                stride_cnn_activations.append(stride_cnn)
                print('final stride cnn layer %d: ' % i, stride_cnn)
                
                
        print('stride cnn (encoder output)', stride_cnn)
        stride_steps = num_splits
        for cnn_layer_spec in cnn_spec:
            stride_steps = stride_steps if not cnn_layer_spec['max_pool'] else int(stride_steps / cnn_layer_spec['max_pool'])
            stride_steps = int(stride_steps / cnn_layer_spec['stride'])
              
        if use_deconv:
            with tf.variable_scope('stride_cnn_decoder_structures'):
                with tf.device('/gpu:0'):

                    stride_cnn = tf.reshape(stride_cnn, (-1, 1, stride_steps, cnn_spec[-1]['filters']))
                    print('stride_cnn encoder output reshaped: ', stride_cnn)

                    for i, cnn_layer_spec in zip(range(len(cnn_spec) - 1, -1, -1), cnn_spec[::-1]):
                        f = cnn_layer_spec['filters']
                        pl = cnn_layer_spec['max_pool']
                        ks = cnn_layer_spec['kernel_size']
                        if not pl:
                            continue


                        stride_cnn = tf.layers.Conv2DTranspose(filters=f, kernel_size=[1,ks], strides=[1,pl], padding='same',
                                                              data_format='channels_last', activation=tf.nn.leaky_relu,
                                                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                                      dtype=tf.float64),
                                                              name='deconv-%d' % i)(stride_cnn)
                        print('deconv-%d: ' % i, stride_cnn)
                    # create last deconv layer
                    fd_strides = int(timesteps/num_splits)
                    stride_cnn = tf.layers.Conv2DTranspose(filters=1, kernel_size=[1,ks], strides=[1,fd_strides], padding='same',
                                                           data_format='channels_last', activation=tf.nn.leaky_relu,
                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False,
                                                                                                           dtype=tf.float64),
                                                           name='deconv-to-input')(stride_cnn)
                    
                    print('final stride_cnn: ', stride_cnn)
            final_stride_cnn_output = tf.reshape(stride_cnn, (-1, timesteps))

        else:
            print('skipping deconv and reshaping')
            final_stride_cnn_output = tf.reshape(stride_cnn, (-1, int(stride_steps * cnn_spec[-1]['filters'])))
            
        # final_stride_cnn_output = tf.reshape(stride_cnn, (-1, int(stride_steps * 4)))
        print('reshaped stride cnn output', final_stride_cnn_output)
        
        # Regularization
        if regularize_networks:
            with tf.variable_scope('stride_cnn_loss'):
                stride_cnn_dense_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(final_stride_cnn_output)
                print('stride_cnn_dense_out')
                stride_cnn_reg_loss = tf.reduce_sum(tf.abs(target - stride_cnn_dense_out))
                loss += stride_cnn_reg_loss
                stride_cnn_reg_op = tf.metrics.mean_absolute_error(labels=target,
                                                                   predictions=stride_cnn_dense_out,
                                                                   name='cnn_mae_op')
                tf.summary.scalar('stride_cnn_mae', stride_cnn_reg_op[1])
                eval_metrics.update({'stride_cnn_regularization/stride_cnn_mae': stride_cnn_reg_op})
    
    if use_stft_cnn:
        with tf.variable_scope('stft_cnn_input'):
            stft_cnn_input = tf.reshape(input_layer, [-1, timesteps])
            stft_cnn_input = tf.contrib.signal.stft(stft_cnn_input, frame_length=stft_frame_length,
                                                    frame_step=stft_frame_step, pad_end='None', name='stft_cnn_op')
            stft_cnn_input = tf.real(stft_cnn_input * tf.conj(stft_cnn_input))
            log_offset = 1e-6
            stft_cnn_input = tf.log(stft_cnn_input + log_offset)  # take log power, as is tradition
            stft_cnn_input = tf.cast(stft_cnn_input, tf.float64)
            tf.summary.histogram('stft_cnn_input', stft_cnn_input)
            print('stft_cnn_input', stft_cnn_input)
        
        # Create CNN layers
        with tf.variable_scope('stft_cnn_structures'):
            with tf.device('/gpu:0'):
                stft_cnn = stft_cnn_input
                for i, cnn_layer_spec in enumerate(cnn_spec):  # reuse CNN
                    s = cnn_layer_spec
                    stft_cnn = tf.layers.Conv1D(filters=s['filters'], kernel_size=s['kernel_size'],
                                                padding='same', activation=tf.nn.leaky_relu, dtype=tf.float64)(stft_cnn)
                    if s['batch_norm']:
                        stft_cnn = tf.layers.batch_normalization(stft_cnn,
                                                                 training=mode == tf.estimator.ModeKeys.TRAIN)
                    if s['max_pool']:
                        stft_cnn = tf.layers.MaxPooling1D(2, 2)(stft_cnn)
                    if mode == tf.estimator.ModeKeys.TRAIN and s['dropout']:
                        stft_cnn = tf.layers.Dropout(rate=dropout_rate)(stft_cnn)
                    print('cnn layer %d: %s' % (i, stft_cnn))
                
                
        print('stft cnn (output)', stft_cnn)
        stft_steps = int(timesteps / stft_frame_step)
        for cnn_layer_spec in cnn_spec:
            stft_steps = stft_steps if cnn_layer_spec['max_pool'] == 0 else int(stft_steps / 2)
        final_stft_cnn_output = tf.reshape(stft_cnn, (-1, int(stft_steps * cnn_spec[-1]['filters'])))
        print('reshaped stft cnn output', final_stft_cnn_output)
        
        # Regularization
        if regularize_networks:
            with tf.variable_scope('stft_cnn_loss'):
                stft_cnn_dense_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(final_stft_cnn_output)
                print('stft_cnn_dense_out')
                stft_cnn_reg_loss = tf.reduce_sum(tf.abs(target - stft_cnn_dense_out))
                loss += stft_cnn_reg_loss
                stft_cnn_reg_op = tf.metrics.mean_absolute_error(labels=target,
                                                                 predictions=stft_cnn_dense_out,
                                                                 name='stft_cnn_mae_op')
                tf.summary.scalar('stft_cnn_mae', stft_cnn_reg_op[1])
                eval_metrics.update({'stft_cnn_regularization/stft_cnn_mae': stft_cnn_reg_op})
    
    
    # This code takes maxval of every time_pool steps to compress the input sequence. it didn't work very well.
    if use_stride:
        with tf.variable_scope('stride_rnn_input'):
            # take max absolute val every time_pool steps
            # for 4096 input w/ time_pool=8, we take abs max of every 8 timesteps for a resulting dimension of (batch_size, 4096/8=512)
            with tf.device('/gpu:0'):
                num_splits = int(timesteps/time_pool)
                stride_rnn_input = tf.cast(to_timepool(input_layer, timesteps, time_pool), tf.float64)
                stride_rnn_input = tf.transpose(stride_rnn_input, [1, 0, 2])                                #(num_splits, batch_size, 1)
                print('stride_rnn_input', stride_rnn_input)

        # Create LSTM layers
        with tf.variable_scope('stride_lstm', reuse=tf.AUTO_REUSE):
            with tf.device('/gpu:0'):
                stride_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(lstm_cell_size),
                                                             lstm_cell_size[-1],
                                                             dropout=dropout_rate,
                                                             direction=lstm_directionality,
                                                             dtype=tf.float64)
                stride_outputs, stride_states = stride_lstm(stride_rnn_input)
            tf.summary.histogram('stride_lstm_states', stride_states)

        print('stride_outputs', stride_outputs)
        final_stride_outputs = tf.transpose(stride_outputs, [1, 0, 2])
        tf.summary.histogram('final_stride_outputs', final_stride_outputs)
        
        # Regularization network for stride network
        if regularize_networks:
            with tf.variable_scope('stride_rnn_regularization'):
                stride_reg_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(final_stride_outputs[:,-1,:])
                stride_reg_out = tf.squeeze(stride_reg_out, axis=1)
                print('stride_reg_out', stride_reg_out)
                stride_reg_loss = tf.reduce_sum(tf.abs(target - stride_reg_out))
                loss += stride_reg_loss
                stride_mae_op = tf.metrics.mean_absolute_error(labels=target,
                                                               predictions=stride_reg_out,
                                                               name='stride_mae_op')
                tf.summary.scalar('stride_mae', stride_mae_op[1])
                eval_metrics.update({'stride_rnn_regularization/stride_mae': stride_mae_op})
        if use_attention:
            with tf.variable_scope('stride_rnn_attention_decoder'):
                # build score network
                N_inp = lstm_cell_size[-1]
                stride_decoder_w = tf.get_variable('stride_attn_w', shape=(N_inp * 2, attn_size), dtype=tf.float64,
                                                   initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1, dtype=tf.float64))
                stride_decoder_b = tf.get_variable('stride_attn_b', shape=(N_inp * 2,), dtype=tf.float64,
                                                   initializer=tf.initializers.zeros(dtype=tf.float64))
                stride_decoder_v = tf.get_variable('stride_attn_v', shape=(attn_size, 1), dtype=tf.float64,
                                                   initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01, dtype=tf.float64))
                def get_context(enc_rnn_states, dec_rnn_state):
                    scores = []
                    for i in range(num_splits):
                        attn_score_inp = tf.concat([enc_rnn_states[:,i,:], dec_rnn_state], axis=1)
                        score = tf.matmul(tf.math.tanh(tf.matmul(attn_score_inp + stride_decoder_b, stride_decoder_w)), stride_decoder_v)
                        scores.append(score)
                    scores_tensor = tf.stack(scores, axis=1)                       # [batch_size, num_splits, 1]
                    #return tf.nn.softmax(scores_tensor, axis=1) * enc_rnn_states   # [batch_size, num_splits, lstm_cell_size[-1]]
                    # Need to multiply weights by scores to get [batch_size, lstm_cell_size] as retval
                
                attn_cells = [build_dropout_lstm_cell(num_units=lstm_cell_size[i],
                                                      activation=tf.nn.leaky_relu,
                                                      reuse=False,
                                                      dtype=tf.float64,
                                                      dropout_rate=dropout_rate,
                                                      name='lstm_cell_%d' % i,
                                                      mode=mode) for i in range(len(lstm_cell_size))]
                attn_net = tf.nn.rnn_cell.MultiRNNCell(attn_cells)
                attn_states = attn_net.zero_state(batch_size)
                attn_input = final_stride_ouputs[:,-1,:]
                # build static attention network (no dynamic unrolling)
                for i in range(attn_steps):
                    # run score network
                    for cell_num in range(len(lstm_cell_size)):
                        attn_states[cell_num][0][1] = get_context(stride_states, attn_states[cell_num][0][1])
                    attn_outputs, attn_states = attn_net(attn_input, attn_states)
                    attn_input = attn_outputs
                
                final_stride_outputs = attn_outputs
    
    
    # more features: STFT
    if use_stft:
        with tf.variable_scope('stft_rnn_input'):
            stft_rnn_input = tf.reshape(input_layer, [-1, timesteps])
            stft_rnn_input = tf.contrib.signal.stft(stft_rnn_input, frame_length=stft_frame_length,
                                                    frame_step=stft_frame_step, pad_end='None', name='stft_op')
            stft_rnn_input = tf.real(stft_rnn_input * tf.conj(stft_rnn_input))
            log_offset = 1e-6
            stft_rnn_input = tf.log(stft_rnn_input + log_offset)  # take log power, as is tradition
            stft_rnn_input = tf.transpose(stft_rnn_input, [1, 0, 2])
            stft_rnn_input = tf.cast(stft_rnn_input, tf.float64)
            tf.summary.histogram('stft_rnn_input', stft_rnn_input)
            print('stft_rnn_input', stft_rnn_input)

        # Create LSTM layers
        with tf.variable_scope('stft_lstm', reuse=tf.AUTO_REUSE):
            with tf.device('/gpu:0'):
                stft_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(len(lstm_cell_size),
                                                               lstm_cell_size[-1],
                                                               dropout=dropout_rate,
                                                               direction=lstm_directionality,
                                                               dtype=tf.float64)
                stft_outputs, stft_states = stft_lstm(stft_rnn_input)
            tf.summary.histogram('stft_lstm_states', stft_states)

        print('stft_outputs', stft_outputs)
        final_stft_outputs = tf.transpose(stft_outputs, [1, 0, 2])
        tf.summary.histogram('final_stft_outputs', final_stft_outputs)
        
        if regularize_networks:
            # Regularization network for stft network
            with tf.variable_scope('stft_rnn_regularization'):
                stft_reg_out = tf.layers.Dense(units=1, activation=None, dtype=tf.float64)(final_stft_outputs[:,-1,:])
                stft_reg_out = tf.squeeze(stft_reg_out, axis=1)
                print('stft_reg_out', stft_reg_out)
                stft_reg_loss = tf.reduce_sum(tf.abs(target - stft_reg_out))
                loss += stft_reg_loss
                stft_mae_op = tf.metrics.mean_absolute_error(labels=target,
                                                             predictions=stft_reg_out,
                                                             name='stft_mae_op')
                tf.summary.scalar('stft_mae', stft_mae_op[1])
                eval_metrics.update({'stft_rnn_regularization/stft_mae': stft_mae_op})
    
    
    # calculate expected shape
    if use_stft:
        stft_steps = int(timesteps / stft_frame_step)
        stft_units = directionality_factor * lstm_cell_size[-1] * stft_steps
        print('total stft elems per batch:', stft_units)
        stack_stft = tf.reshape(final_stft_outputs, (-1, stft_units))
    if use_stride:
        stride_steps = num_splits
        stride_units = directionality_factor * lstm_cell_size[-1] * stride_steps
        print('total stride elems per batch:', stride_units)
        stack_stride = tf.reshape(final_stride_outputs, (-1, stride_units))
    
        # or, if you only want to use the last state:
        stack_stride = tf.reshape(final_stride_outputs[:,-1,:], (-1, int(stride_units/stride_steps)))
    concats = []
    if use_stride_cnn:
        concats.append(final_stride_cnn_output)
    if use_stft_cnn:
        concats.append(final_stft_cnn_output)
    if use_stft:
        concats.append(stack_stft)
    if use_stride:
        concats.append(stack_stride)
    
    if not use_stride_cnn and not use_stft and not use_stride and not use_stft_cnn:
        print('WARNING: No features except input layers are selected!')
        concats.append(input_layer)  # model with no intermediate steps
    
    dense = tf.concat(concats, axis=1)
    print('dense input', dense)
    
    # Pass through a Dense layer(s)
    
    if use_dense_head:
        dense_layers = []
        with tf.variable_scope('dense_head'):
            for i, dense_sz in enumerate(dense_size):
                dense = tf.layers.Dense(units=dense_sz,
                                        activation=tf.nn.leaky_relu,
                                        name='dense_layer_%d' % i,
                                        dtype=tf.float64)(dense)
                print('dense %d: ' % i, dense)

                if dense_batch_norm:
                    if i < len(dense_size) - 1:
                        dense = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN)

                dense_layers.append(dense)  # Track because we need to return these for prediction (features!)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    dense = tf.layers.Dropout(rate=dropout_rate)(dense)

            dense = tf.layers.Dense(units=1,
                                    activation=None,
                                    name='dense_layer_out',
                                    dtype=tf.float64)(dense)
        # Reshape dense outputs to same shape as `labels'
        results = tf.squeeze(dense, axis=1)
    
    else:
        results = dense
    
    print('results', results)
    # print('_labels', _labels)
    print('labels', labels)
    print('target', target)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred': results,
            # 'dense_feats': dense_layers[-2]
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, prediction_hooks=[SlimVarAnalyzer()])
    
        
    with tf.variable_scope('loss_sse'):
        # calculate sum of squared errors
        loss += tf.reduce_sum(tf.square(target[:,-1] - results), name='loss_sse')
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
    mse_op = tf.metrics.mean_squared_error(labels=target[:,-1], #labels=input_feats[:,-1]
                                           predictions=results,
                                           name='mse_op')
    mae_op = tf.metrics.mean_absolute_error(labels=target[:,-1], #labels=input_feats[:,-1]
                                            predictions=results,
                                            name='mae_op')
    tf.summary.scalar('mse', mse_op[1])
    tf.summary.scalar('mae', mae_op[1])
    eval_metrics.update({'mse': mse_op, 'mae': mae_op})
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics, evaluation_hooks=[SlimVarAnalyzer()])
    
    
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
        
        if dense_batch_norm:
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
                    tf.summary.histogram('VAR_' + str(y), y)
                    tf.summary.histogram('GRAD_' + str(y), x)
                clipped = [tf.cast(g, grad_dtypes[i]) for i, g in enumerate(clipped)]
                train_op = optimizer.apply_gradients(zip(clipped, tfvars), global_step=tf.train.get_global_step())
                #train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        else:
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
                tf.summary.histogram('VAR_' + str(y), y)
                tf.summary.histogram('GRAD_' + str(y), x)
            clipped = [tf.cast(g, grad_dtypes[i]) for i, g in enumerate(clipped)]
            train_op = optimizer.apply_gradients(zip(clipped, tfvars), global_step=tf.train.get_global_step())
        
            
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[SlimVarAnalyzer()])
