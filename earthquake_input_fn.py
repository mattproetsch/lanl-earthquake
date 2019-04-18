# uncompyle6 version 3.2.5
# Python bytecode 3.5 (3350)
# Decompiled from: Python 3.5.2 (default, Nov 23 2017, 16:37:01) 
# [GCC 5.4.0 20160609]
# Embedded file name: /workspace/persistent-code/earthquake_input_fn.py
# Compiled at: 2019-03-07 08:36:50
# Size of source mod 2**32: 2730 bytes
import tensorflow as tf, os.path as osp
from glob import glob
import os, numpy as np
from time import time
from freq_preprocess import to_stft, to_pca, fft_pca
from stat_preprocess import to_timepool

def _deserialize_earthquakes2(serialized_examples, timesteps, window_shift, training, noise):
    N_PER_FILE = 150000
    BUFFER = 0
    LEN = N_PER_FILE + BUFFER
    features = {
        'acousticdata': tf.FixedLenFeature(LEN, dtype=tf.int64), 
        'tminus': tf.FixedLenFeature(LEN, dtype=tf.string)
    }
    features = tf.io.parse_single_example(serialized_examples, features=features)
    features['tminus'] = tf.strings.to_number(string_tensor=features['tminus'], out_type=tf.float64)
    features['acousticdata'] = tf.cast(features['acousticdata'], tf.float32)
    
    acousticdata = features['acousticdata'] / 5515.0
    tminus = features['tminus'] / 16.1
    
    if training and noise > 0.001:
        acousticdata = tf.clip_by_value(acousticdata * tf.random.normal((LEN,), mean=1, stddev=noise ** 2) + tf.random.normal((LEN,), mean=0, stddev=noise), acousticdata - noise * 1.96, acousticdata + noise * 1.96)
    
    acoustic_stat = to_timepool(acousticdata, LEN, window_shift, tf.float32)
    print('acoustic_stat', acoustic_stat)
    freq_func = fft_pca(timesteps, 256, window_shift, tf.float32)
    acoustic_freq = tf.py_func(freq_func, (features['acousticdata'][window_shift:],), tf.float32)
    acoustic_freq = tf.reshape(acoustic_freq, [int(timesteps/window_shift), 16])
    print('acoustic_freq after py_func', acoustic_freq)
    cc = tf.concat([acoustic_stat, acoustic_freq], axis=1)
    print('catted', cc)
    
    tminus = tf.reshape(tminus[::window_shift], (-1, 1))
    tminus = tf.tile(tminus, (1, 66))
    tminus = tf.reshape(tminus, (-1,))
    cc = tf.reshape(cc, (-1,))
    print('tminus', tminus)
    print('cc', cc)
    tf.print('tminus', tminus)
    tf.print('cc', cc)
    dataset = tf.data.Dataset.from_tensor_slices(({'acousticdata': cc}, tminus))
    dataset = dataset.batch(int(timesteps/window_shift) * 66, drop_remainder=True).take(1)
    return dataset


def earthquake_input_fn2(basedir, batch_size, timesteps, noise=0.001, window_shift=None, traintest='none', epochs=1, eager=False, seed=1234):
    if not traintest in ('train', 'test'):
        raise AssertionError('must specify traintest as "train" or "test"')
    if not 150000 % timesteps == 0:
        # raise AssertionError('timesteps should divide 150k otherwise your test data is messed up')
        pass
    files = glob(osp.join(basedir, traintest, '*.tfrecord'))
    if traintest == 'train':
        np.random.seed(np.int64(time()))
    else:
        np.random.seed(np.int64(seed))
    files = np.random.permutation(files)
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if traintest == 'train':
        dataset = dataset.repeat(epochs).shuffle(1000)
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=16)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(lambda x: _deserialize_earthquakes2(x,
                                                                                                         timesteps,
                                                                                                         window_shift,
                                                                                                         traintest == 'train',
                                                                                                         noise),
                                                                     cycle_length=16, block_length=1, sloppy=True))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(100)
    if eager:
        return dataset
    data_iter = dataset.make_one_shot_iterator()
    return data_iter.get_next()
# okay decompiling earthquake_input_fn.cpython-35.pyc
