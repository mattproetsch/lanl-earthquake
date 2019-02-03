import tensorflow as tf
import os.path as osp
from glob import glob
import os
import numpy as np
from time import time

def _deserialize_earthquakes2(serialized_examples, timesteps, training):
    N_PER_FILE = 1500000
    features = {
        'acousticdata': tf.FixedLenFeature((N_PER_FILE), dtype=tf.int64),
        'tminus': tf.FixedLenFeature((N_PER_FILE), dtype=tf.string)
    }
    features = tf.io.parse_single_example(
        serialized_examples,
        features=features
    )
    
    features['tminus'] = tf.strings.to_number(string_tensor=features['tminus'], out_type=tf.float64)
    features['acousticdata'] = tf.cast(features['acousticdata'], tf.float32)
    
    dataset = tf.data.Dataset.from_tensor_slices(({'acousticdata': features['acousticdata'] / 5515.0}, features['tminus'] / 16.10))
    if training:
        if np.random.random() > 0.75:  # with 75% chance, skip some number of rows at the beginning of the file
            dataset = dataset.skip(np.random.randint(timesteps))
    
    dataset = dataset.batch(timesteps, drop_remainder=True)
    
    if training:
        dataset = dataset.shuffle(int(N_PER_FILE / timesteps))
    
    return dataset

def earthquake_input_fn2(basedir, batch_size, timesteps, window_shift=None, traintest='none', eager=False, seed=1234):
    
    assert traintest in ['train', 'test'], 'must specify traintest as "train" or "test"'
    assert 150000 % timesteps == 0, 'timesteps should divide 150k otherwise your test data is messed up'
    
    files = glob(osp.join(basedir, traintest, '*.tfrecord'))
    
    if traintest == 'train':
        np.random.seed(np.int64(time()))
    else:
        np.random.seed(np.int64(seed))  # deterministic seed for consistent eval
        
    files = np.random.permutation(files)
        
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=8)
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(\
                            lambda x: _deserialize_earthquakes2(x, timesteps, traintest == 'train'),
                            cycle_length=8,
                            block_length=4))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(100)
    
    if eager:
        return dataset
    
    data_iter = dataset.make_one_shot_iterator()
    return data_iter.get_next()

