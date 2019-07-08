import numpy as np
import librosa
from tqdm import tqdm

# disabling uncomfortable warnings
from absl import logging
logging._warn_preinit_stderr = 0
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# setting the path
import sys
sys.path.append("./musically_motivated_CNN/")

# importing musiCNN models
import models
import configuration as config


def batch_data(audio_file, n_frames, overlap):

    # compute the log-mel spectrogram with librosa
    audio, sr = librosa.load(audio_file, sr=config.SR)
    audio_rep = librosa.feature.melspectrogram(y=audio, 
                                               sr=sr,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE,
                                               n_mels=config.N_MELS).T
    audio_rep = audio_rep.astype(np.float16)
    audio_rep = np.log10(10000 * audio_rep + 1)

    # batch it for an efficient computing
    first = True
    last_frame = audio_rep.shape[0] - n_frames + 1
    # +1 is to include the last frame that range would not include
    for time_stamp in tqdm(range(0, last_frame, overlap)):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


def predict(file_name, model='MTT', in_length=3, in_overlap=None, features=None):

    # select model
    if model == 'MTT':
        labels = config.MTT_LABELS
    elif model == 'MSD':
        labels = config.MSD_LABELS
    num_classes = len(labels)

    # convert seconds to frames
    n_frames = librosa.time_to_frames(in_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    if not in_overlap:
        overlap = n_frames
    else:
        overlap = librosa.time_to_frames(in_overlap, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)

    # tensorflow: define the model
    with tf.name_scope('model'):
        x = tf.placeholder(tf.float32, [None, n_frames, config.N_MELS])
        is_training = tf.placeholder(tf.bool)
        y, summarized_features, features = models.define_models(x, is_training, model, num_classes)
        normalized_y = tf.nn.sigmoid(y)

    # tensorflow: loading model
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './musically_motivated_CNN/' + model + '/') 

    # batching data
    print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..')
    batch, spectrogram = batch_data(file_name, n_frames, overlap)

    # tensorflow: extract features and tags
    # ..first batch!
    predicted_tags, patch_embedding, tmp_emb = sess.run([normalized_y, summarized_features, features], 
                                                               feed_dict={x: batch[:config.BATCH_SIZE], 
                                                               is_training: False})
    taggram = np.array(predicted_tags)
    full_embedding = np.squeeze(tmp_emb)

    # ..rest of the batches!
    for id_pointer in tqdm(range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE)):
        predicted_tags, tmp_downsampled_emb, tmp_emb = sess.run([normalized_y, summarized_features, features], 
                                                                 feed_dict={x: batch[id_pointer:id_pointer+config.BATCH_SIZE], 
                                                                 is_training: False})
        taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)
        patch_embedding = np.concatenate((patch_embedding, tmp_downsampled_emb), axis=0)
        full_embedding = np.concatenate((full_embedding, np.squeeze(tmp_emb)), axis=0)

    return taggram, labels#, patch_embedding, full_embedding, np.float32(spectrogram)


