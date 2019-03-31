import tensorflow as tf

def extract_stats(tens, axnum):
    """ returns (min, max, mean, var) along axis axnum """
    tens_min = tf.reduce_min(tens, axis=axnum)
    tens_max = tf.reduce_max(tens, axis=axnum)
    tens_mean, tens_var = tf.nn.moments(tens, axes=[axnum])
    return tens_min, tens_max, tens_mean, tens_var


def extract_pctiles(tens, *pctiles):
    return tf.concat([tf.contrib.distributions.percentile(tens, p, axis=2, keep_dims=True) for p in pctiles], axis=2)


def to_stft(tens, timesteps, window_size, window_step):
    stft_input = tf.reshape(tens, [-1, timesteps])
    stft_input = tf.contrib.signal.stft(stft_input, frame_length=window_size,
                                        frame_step=window_step, pad_end='None', name='stft_op')
    stft_input = tf.real(stft_input * tf.conj(stft_input))
    log_offset = 1e-6
    stft_input = tf.log(stft_input + log_offset)  # take log power, as is tradition
    stft_input = tf.cast(stft_input, tf.float64)
    tf.summary.histogram('stft_input', stft_input)
    print('stft_input', stft_input)
    return stft_input


def to_timepool(tens, timesteps, time_pool):
    """
    take pooled features every time_pool steps 
    for 4096 input w/ time_pool=8, we take abs max of every 8 timesteps for a resulting dimension of (batch_size, 4096/8=512)
    """
    
    stride_input = tf.reshape(tens, [-1, timesteps])
    num_splits = int(timesteps/time_pool)
    stride_input = tf.cast(tens, tf.float64)
    print('num_splits=%d (this is the number of timesteps fed to stride CNN)' % num_splits)
    stride_input = tf.split(stride_input, num_or_size_splits=num_splits, axis=1)        #([batch_size, time_pool] * num_splits)
    stride_input = tf.stack(stride_input, axis=1)                                       #(batch_size, num_splits, time_pool))
    
    stride_min, stride_max, stride_mean, stride_var = extract_stats(stride_input, axnum=2)
    stride_pctiles = extract_pctiles(stride_input, 1, 5, 25, 50, 75, 99)
    
    stride_roc = stride_input[:,:,1:] - stride_input[:,:,:-1]
    stride_minroc, stride_maxroc, stride_meanroc, stride_varroc = extract_stats(stride_roc, axnum=2)
    stride_roc_pctiles = extract_pctiles(stride_roc, 1, 5, 25, 50, 75, 99)
    
    stride_roroc = stride_roc[:,:,1:] - stride_roc[:,:,:-1]
    stride_minroroc, stride_maxroroc, stride_meanroroc, stride_varroroc = extract_stats(stride_roroc, axnum=2)
    stride_roroc_pctiles = extract_pctiles(stride_roroc, 1, 5, 25, 50, 75, 99)
    
    stride_skew = stride_roroc[:,:,1:] - stride_roroc[:,:,:-1]
    skew_min, skew_max, skew_mean, skew_var = extract_stats(stride_skew, axnum=2)
    stride_skew_pctiles = extract_pctiles(stride_skew, 1, 5, 25, 50, 75, 99)
    
    stride_kurt = stride_skew[:,:,1:] - stride_skew[:,:,:-1]
    kurt_min, kurt_max, kurt_mean, kurt_var = extract_stats(stride_kurt, axnum=2)
    kurt_pctiles = extract_pctiles(stride_kurt, 1, 5, 25, 50, 75, 99)
    
    stft = to_stft(tens, timesteps, time_pool, time_pool)
    _, top_freq_indices = tf.math.top_k(stft, k=3)
    freq_embeddings = tf.get_variable('frequency_embeddings', [stft.shape[-1], 8], dtype=tf.float64)
    top_freq_embeddings = tf.nn.embedding_lookup(freq_embeddings, top_freq_indices)
    top_freq_embeddings = tf.reshape(top_freq_embeddings, [-1, num_splits, top_freq_indices.shape[-1] * freq_embeddings.shape[-1]])
    
    print('stride_min', stride_min)
    print('stride_max', stride_max)
    print('stride_mean', stride_mean)
    print('stride_var', stride_var)
    print('stride_pctiles', stride_pctiles)
    print('stride_minroc', stride_minroc)
    print('stride_maxroc', stride_maxroc)
    print('stride_meanroc', stride_meanroc)
    print('stride_varroc', stride_varroc)
    print('stride_roc_pctiles', stride_roc_pctiles)
    print('stride_minroroc', stride_minroroc)
    print('stride_maxroroc', stride_maxroroc)
    print('stride_meanroroc', stride_meanroroc)
    print('stride_varroroc', stride_varroroc)
    print('stft', stft)
    print('top_freq_indices', top_freq_indices)
    print('top_freq_embeddings', top_freq_embeddings)
    
    stride_input = tf.stack([stride_min, stride_max, stride_mean, stride_var,
                             *tf.unstack(stride_pctiles, axis=2),
                             stride_minroc, stride_maxroc, stride_meanroc, stride_varroc,
                             *tf.unstack(stride_roc_pctiles, axis=2),
                             stride_minroroc, stride_maxroroc, stride_meanroroc, stride_varroroc,
                             *tf.unstack(stride_roc_pctiles, axis=2),
                             skew_min, skew_max, skew_mean, skew_var,
                             *tf.unstack(stride_skew_pctiles, axis=2),
                             kurt_min, kurt_max, kurt_mean, kurt_var,
                             *tf.unstack(kurt_pctiles, axis=2),
                             *tf.unstack(top_freq_embeddings, axis=2),
                            ], axis=2)
    return stride_input