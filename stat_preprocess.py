import tensorflow as tf

def extract_stats(tens, axnum):
    """ returns (min, max, mean, var) along axis axnum """
    tens_min = tf.reduce_min(tens, axis=axnum)
    tens_max = tf.reduce_max(tens, axis=axnum)
    tens_mean, tens_var = tf.nn.moments(tens, axes=[axnum])
    return tens_min, tens_max, tens_mean, tens_var


def extract_pctiles(tens, *pctiles):
    return tf.concat([tf.contrib.distributions.percentile(tens, p, axis=1, keep_dims=True) for p in pctiles], axis=1)


def to_timepool(tens, timesteps, time_pool, dtype):
    """
    take pooled features every time_pool steps 
    for 4096 input w/ time_pool=8, we take abs max of every 8 timesteps for a resulting dimension of (batch_size, 4096/8=512)
    """

    stride_input = tf.reshape(tens, [timesteps])
    num_splits = int(timesteps/time_pool)
    stride_input = tf.cast(tens, dtype)
    
    def order_split(tens, order):
        tens = tf.pad(tens, [[order, 0]])
        tens = tf.split(tens, num_or_size_splits=num_splits, axis=0)
        tens = tf.stack(tens, axis=0)
        return tens
    
    stride_input_o1 = stride_input
    stride_input_o2 = stride_input_o1[1:] - stride_input_o1[:-1]
    stride_input_o3 = stride_input_o2[1:] - stride_input_o2[:-1]
    stride_input_o4 = stride_input_o3[1:] - stride_input_o3[:-1]
    stride_input_o5 = stride_input_o4[1:] - stride_input_o4[:-1]
    
    stride_input_o1 = order_split(stride_input_o1, 0)
    stride_input_o2 = order_split(stride_input_o2, 1)
    stride_input_o3 = order_split(stride_input_o3, 2)
    stride_input_o4 = order_split(stride_input_o4, 3)
    stride_input_o5 = order_split(stride_input_o5, 4)
    
    print('stride_input_o1', stride_input_o1)
    print('stride_input_o2', stride_input_o2)
    print('stride_input_o3', stride_input_o3)
    print('stride_input_o4', stride_input_o4)
    print('stride_input_o5', stride_input_o5)

    print('num_splits=%d (this is the number of timesteps fed to stride CNN)' % num_splits)

    stride_o1_min, stride_o1_max, stride_o1_mean, stride_o1_var = extract_stats(stride_input_o1, axnum=1)
    stride_o1_pctiles = extract_pctiles(stride_input_o1, 1, 5, 25, 50, 75, 99)
    
    stride_o2_min, stride_o2_max, stride_o2_mean, stride_o2_var = extract_stats(stride_input_o2, axnum=1)
    stride_o2_pctiles = extract_pctiles(stride_input_o2, 1, 5, 25, 50, 75, 99)
    
    stride_o3_min, stride_o3_max, stride_o3_mean, stride_o3_var = extract_stats(stride_input_o3, axnum=1)
    stride_o3_pctiles = extract_pctiles(stride_input_o3, 1, 5, 25, 50, 75, 99)
    
    stride_o4_min, stride_o4_max, stride_o4_mean, stride_o4_var = extract_stats(stride_input_o4, axnum=1)
    stride_o4_pctiles = extract_pctiles(stride_input_o4, 1, 5, 25, 50, 75, 99)
    
    stride_o5_min, stride_o5_max, stride_o5_mean, stride_o5_var = extract_stats(stride_input_o5, axnum=1)
    stride_o5_pctiles = extract_pctiles(stride_input_o5, 1, 5, 25, 50, 75, 99)

    stride_input = tf.stack([stride_o1_min, stride_o1_max, stride_o1_mean, stride_o1_var,
                             *tf.unstack(stride_o1_pctiles, axis=1),
                             stride_o2_min, stride_o2_max, stride_o2_mean, stride_o2_var,
                             *tf.unstack(stride_o2_pctiles, axis=1),
                             stride_o3_min, stride_o3_max, stride_o3_mean, stride_o3_var,
                             *tf.unstack(stride_o3_pctiles, axis=1),
                             stride_o4_min, stride_o4_max, stride_o4_mean, stride_o4_var,
                             *tf.unstack(stride_o4_pctiles, axis=1),
                             stride_o5_min, stride_o5_max, stride_o5_mean, stride_o5_var,
                             *tf.unstack(stride_o5_pctiles, axis=1)
                            ], axis=1)
    print('stride_input', stride_input)
    return stride_input