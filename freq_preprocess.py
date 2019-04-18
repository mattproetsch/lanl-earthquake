import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import joblib
from scipy.signal import stft

pca = joblib.load('/workspace/persistent-code/pca.skmodel')

def to_stft(tens, timesteps, window_size, window_step, dtype):
    stft_input = tf.reshape(tens, [timesteps])
    stft_input = tf.contrib.signal.stft(stft_input, frame_length=window_size,
                                        frame_step=window_step, fft_length=256, pad_end='None', name='stft_op')
    stft_input = tf.real(stft_input * tf.conj(stft_input))
    # log_offset = 1e-6
    #stft_input = tf.log(stft_input + log_offset)  # take log power, as is tradition
    stft_input = tf.cast(stft_input, dtype)
    # tf.summary.histogram('stft_input', stft_input)
    print('stft_input', stft_input)
    return stft_input


# def pca_fft(tens, fftlen, overlap, dtype):
#     stft = to_stft(tens, fftlet, fftlen - overlap, dtype)
#     _, top_freq_indices = tf.math.top_k(stft, k=3)
#     freq_embeddings = tf.get_variable('frequency_embeddings', [stft.shape[-1], 8], dtype=dtype)
#     top_freq_embeddings = tf.nn.embedding_lookup(freq_embeddings, top_freq_indices)
#     top_freq_embeddings = tf.reshape(top_freq_embeddings, [-1, num_splits, top_freq_indices.shape[-1] * freq_embeddings.shape[-1]])

def to_pca(nparr):
    return pca.transform(nparr)

def fft_pca(timesteps, window_size, window_step, dtype):
    #print('fft_pca', timesteps, window_size, window_step, dtype)
    def do_func(nparr):
        #print('do_func: nparr', nparr.shape)
        _, _, fqs = stft(nparr, nperseg=window_size, noverlap=window_size-window_step)
        #print('do_func: fqs', fqs.shape)
        fqs = np.real(fqs * np.conj(fqs))
        #print('do_func: fqs after reshape', fqs.shape)
        xformed = pca.transform(np.transpose(fqs))
        #print('do_func: xformed', xformed.shape)
        return xformed
    return do_func