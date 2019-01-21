import tensorflow as tf
import os.path as osp
from glob import glob
import os

def _deserialize_earthquakes(serialized_examples):
    features = {
        'acousticdata': tf.FixedLenFeature((4096), tf.int64),
        'tminus': tf.FixedLenFeature((4096), tf.string)
    }
    features = tf.io.parse_single_example(
        serialized_examples,
        features=features
    )
    
    features['tminus'] = tf.strings.to_number(string_tensor=features['tminus'], out_type=tf.float64)
    
    return {'acousticdata': features['acousticdata']}, features['tminus']

def _deserialize_earthquakes2(serialized_examples):
    features = {
        'acousticdata': tf.FixedLenFeature((4096), tf.int64),
        'tminus': tf.FixedLenFeature((4096), tf.string)
    }
    features = tf.io.parse_single_example(
        serialized_examples,
        features=features
    )
    
    features['tminus'] = tf.strings.to_number(string_tensor=features['tminus'], out_type=tf.float64)
    
    return tf.data.Dataset.from_tensor_slices(({'acousticdata': features['acousticdata']}, features['tminus']))


def earthquake_input_fn(basedir, batch_size, timesteps, traintest='none', eager=False):
    assert traintest in ['train', 'test'], 'must specify train or test'
    
    files = glob(osp.join(basedir, traintest, '*.tfrecords'))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=8)
    dataset = dataset.map(_deserialize_earthquakes, num_parallel_calls=8)
    
    if traintest == 'train':
        dataset = dataset.shuffle(1000)
    
    # Turn each input example into a series of 4096/(timesteps) sub-examples
    N_sub_batches = int(4096 / timesteps)
    if N_sub_batches * timesteps != 4096:
        raise Exception('timesteps does not evently divide 4096!')
    
    dataset = dataset.interleave(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)),
                                  cycle_length=batch_size,
                                  block_length=timesteps)
    dataset = dataset.batch(timesteps).batch(batch_size)
    
    dataset = dataset.prefetch(1000)
    
    if eager:
        return dataset
    
    data_iter = dataset.make_one_shot_iterator()
    return data_iter.get_next()

def earthquake_input_fn2(basedir, batch_size, timesteps, scales=None, traintest='none', eager=False):
    """
    `scales` argument should be a list of '1eX', where X can range from -8 to 2
    """
    
    assert traintest in ['train', 'test'], 'must specify traintest as "train" or "test"'
    assert 4096 % timesteps == 0, 'timesteps must evenly divide 4096'
    possible_scales = ['1e%d' % d for d in range(-9, 4+1)]
    
    files = []
    if type(scales) is str:
        scales = [scales]
    for x in scales:
        assert x in possible_scales, 'invalid scale %s, try ["1e0, 1e-2"], "1e1", etc.' % x
        files.extend(glob(osp.join(basedir, traintest, x, '*.tfrecords')))
    
    dataset = tf.data.Dataset.from_tensor_slices(files)
    if traintest == 'train':
        dataset = dataset.shuffle(5000)
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=24)
    
    # Turn each input example into a series of 4096/(timesteps) sub-examples
    N_sub_batches = int(4096 / timesteps)    
    dataset = dataset.apply(tf.data.experimental.parallel_interleave(lambda x: _deserialize_earthquakes2(x),
                                                                     cycle_length=batch_size,
                                                                     block_length=timesteps))
    
    dataset = dataset.batch(timesteps).batch(batch_size)
    dataset = dataset.prefetch(N_sub_batches * 2)
    
    if eager:
        return dataset
    
    data_iter = dataset.make_one_shot_iterator()
    return data_iter.get_next()