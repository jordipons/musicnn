import numpy as np
import librosa
from tqdm import tqdm
import tensorflow as tf
from musiCNN import models
from musiCNN import configuration as config


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


def extractor(file_name, model='MTT', input_length=3, input_overlap=None, extract_features=False):

    # select model
    if model == 'MTT':
        labels = config.MTT_LABELS
    elif model == 'MSD':
        labels = config.MSD_LABELS
    num_classes = len(labels)

    # convert seconds to frames
    n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    if not input_overlap:
        overlap = n_frames
    else:
        overlap = librosa.time_to_frames(input_overlap, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)

    # tensorflow: define the model
    tf.compat.v1.reset_default_graph()
    with tf.name_scope('model'):
        x = tf.compat.v1.placeholder(tf.float32, [None, n_frames, config.N_MELS])
        is_training = tf.compat.v1.placeholder(tf.bool)
        y, timbral, temporal, midend1, midend2, midend3, avg_pool, max_pool, backend = models.define_models(x, is_training, model, num_classes)
        normalized_y = tf.nn.sigmoid(y)

    # tensorflow: loading model
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, './musiCNN/' + model + '/') 

    # batching data
    print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..')
    batch, spectrogram = batch_data(file_name, n_frames, overlap)

    # tensorflow: extract features and tags
    # ..first batch!
    if extract_features:
        extract_vector = [normalized_y, timbral, temporal, midend1, midend2, midend3, avg_pool, max_pool, backend]
    else:
        extract_vector = [normalized_y]

    tf_out = sess.run(extract_vector, 
                      feed_dict={x: batch[:config.BATCH_SIZE], 
                      is_training: False})

    if extract_features:
        predicted_tags, timbral_, temporal_, midend1_, midend2_, midend3_, avg_pool_, max_pool_, backend_ = tf_out
        features = dict()
        features['timbral'] = np.squeeze(timbral_)
        features['temporal'] = np.squeeze(temporal_)
        features['midend1'] = np.squeeze(midend1_)
        features['midend2'] = np.squeeze(midend2_)
        features['midend3'] = np.squeeze(midend3_)
        features['avg_pool'] = avg_pool_
        features['max_pool'] = max_pool_
        features['backend'] = backend_
    else:
        predicted_tags = tf_out[0]

    taggram = np.array(predicted_tags)


    # ..rest of the batches!
    for id_pointer in tqdm(range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE)):

        tf_out = sess.run(extract_vector, 
                          feed_dict={x: batch[id_pointer:id_pointer+config.BATCH_SIZE], 
                          is_training: False})

        if extract_features:
            predicted_tags, timbral_, temporal_, midend1_, midend2_, midend3_, avg_pool_, max_pool_, backend_ = tf_out
            features['timbral'] = np.concatenate((features['timbral'], np.squeeze(timbral_)), axis=0)
            features['temporal'] = np.concatenate((features['temporal'], np.squeeze(temporal_)), axis=0)
            features['midend1'] = np.concatenate((features['midend1'], np.squeeze(midend1_)), axis=0)
            features['midend2'] = np.concatenate((features['midend2'], np.squeeze(midend2_)), axis=0)
            features['midend3'] = np.concatenate((features['midend3'], np.squeeze(midend3_)), axis=0)
            features['avg_pool'] = np.concatenate((features['avg_pool'], avg_pool_), axis=0)
            features['max_pool'] = np.concatenate((features['max_pool'], max_pool_), axis=0)
            features['backend'] = np.concatenate((features['backend'], backend_), axis=0)
        else:
            predicted_tags = tf_out[0]

        taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)

    sess.close()

    if extract_features:
        return taggram, labels, features
    else:
        return taggram, labels

#def top_tags


