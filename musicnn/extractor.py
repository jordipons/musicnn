import os
import numpy as np
import librosa

import tensorflow as tf
# disable eager mode for tf.v1 compatibility with tf.v2
tf.compat.v1.disable_eager_execution()

from musicnn import models
from musicnn import configuration as config


def batch_data(audio_file, n_frames, overlap):
    '''For an efficient computation, we split the full music spectrograms in patches of length n_frames with overlap.

    INPUT
    
    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'

    - n_frames: length (in frames) of the input spectrogram patches.
    Data format: integer.
    Example: 187
        
    - overlap: ammount of overlap (in frames) of the input spectrogram patches.
    Note: Set it considering n_frames.
    Data format: integer.
    Example: 10
    
    OUTPUT
    
    - batch: batched audio representation. It returns spectrograms split in patches of length n_frames with overlap.
    Data format: 3D np.array (batch, time, frequency)
    
    - audio_rep: raw audio representation (spectrogram).
    Data format: 2D np.array (time, frequency)
    '''

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
    for time_stamp in range(0, last_frame, overlap):
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


def extractor(file_name, model='MTT_musicnn', input_length=3, input_overlap=False, extract_features=True):
    '''Extract the taggram (the temporal evolution of tags) and features (intermediate representations of the model) of the music-clip in file_name with the selected model.

    INPUT

    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'
    
    - model: select a music audio tagging model.
    Data format: string.
    Options: 'MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
    MTT models are trained with the MagnaTagATune dataset.
    MSD models are trained with the Million Song Dataset.
    To know more about these models, check our musicnn / vgg examples, and the FAQs.
    Important! 'MSD_musicnn_big' is only available if you install from source: python setup.py install.

    - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
    Note: This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram.
    Recommended value: 3, because the models were trained with 3 second inputs.
    Observation: the vgg models do not allow for different input lengths. For this reason, the vgg models' input_length needs to be set to 3. However, musicnn models allow for different input lengths: see this jupyter notebook.
    Data format: floating point number.
    Example: 3.1
    
    - input_overlap: ammount of overlap (in seconds) of the input spectrogram patches.
    Note: Set it considering the input_length.
    Data format: floating point number.
    Example: 1.0
    
    - extract_features: set it True for extracting the intermediate representations of the model.
    Data format: boolean.
    Options: False (for NOT extracting the features), True (for extracting the features).

    OUTPUT
    
    - taggram: expresses the temporal evolution of the tags likelihood.
    Data format: 2D np.ndarray (time, tags).
    Example: see our basic / advanced examples.
    
    - tags: list of tags corresponding to the tag-indices of the taggram.
    Data format: list.
    Example: see our FAQs page for the complete tags list.
    
    - features: if extract_features = True, it outputs a dictionary containing the activations of the different layers the selected model has.
    Data format: dictionary.
    Keys (musicnn models): ['timbral', 'temporal', 'cnn1', 'cnn2', 'cnn3', 'mean_pool', 'max_pool', 'penultimate']
    Keys (vgg models): ['pool1', 'pool2', 'pool3', 'pool4', 'pool5']
    Example: see our musicnn and vgg examples.

    '''
    
    # select model
    if 'MTT' in model:
        labels = config.MTT_LABELS
    elif 'MSD' in model:
        labels = config.MSD_LABELS
    num_classes = len(labels)
    
    if 'vgg' in model and input_length != 3:
        raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')

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
        if 'vgg' in model:
            y, pool1, pool2, pool3, pool4, pool5 = models.define_model(x, is_training, model, num_classes)
        else:
            y, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate = models.define_model(x, is_training, model, num_classes)
        normalized_y = tf.nn.sigmoid(y)

    # tensorflow: loading model
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    try:
        saver.restore(sess, os.path.dirname(__file__)+'/'+model+'/') 
    except:
        if model == 'MSD_musicnn_big':
            raise ValueError('MSD_musicnn_big model is only available if you install from source: python setup.py install')
        elif model == 'MSD_vgg':
            raise ValueError('MSD_vgg model is still training... will be available soon! :)')

    # batching data
    print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..', end =" ")
    batch, spectrogram = batch_data(file_name, n_frames, overlap)

    # tensorflow: extract features and tags
    # ..first batch!
    if extract_features:
        if 'vgg' in model:
            extract_vector = [normalized_y, pool1, pool2, pool3, pool4, pool5]
        else:
            extract_vector = [normalized_y, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate]
    else:
        extract_vector = [normalized_y]

    tf_out = sess.run(extract_vector, 
                      feed_dict={x: batch[:config.BATCH_SIZE], 
                      is_training: False})

    if extract_features:
        if 'vgg' in model:
            predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
            features = dict()
            features['pool1'] = np.squeeze(pool1_)
            features['pool2'] = np.squeeze(pool2_)
            features['pool3'] = np.squeeze(pool3_)
            features['pool4'] = np.squeeze(pool4_)
            features['pool5'] = np.squeeze(pool5_)
        else:
            predicted_tags, timbral_, temporal_, cnn1_, cnn2_, cnn3_, mean_pool_, max_pool_, penultimate_ = tf_out
            features = dict()
            features['timbral'] = np.squeeze(timbral_)
            features['temporal'] = np.squeeze(temporal_)
            features['cnn1'] = np.squeeze(cnn1_)
            features['cnn2'] = np.squeeze(cnn2_)
            features['cnn3'] = np.squeeze(cnn3_)
            features['mean_pool'] = mean_pool_
            features['max_pool'] = max_pool_
            features['penultimate'] = penultimate_
    else:
        predicted_tags = tf_out[0]

    taggram = np.array(predicted_tags)


    # ..rest of the batches!
    for id_pointer in range(config.BATCH_SIZE, batch.shape[0], config.BATCH_SIZE):

        tf_out = sess.run(extract_vector, 
                          feed_dict={x: batch[id_pointer:id_pointer+config.BATCH_SIZE], 
                          is_training: False})

        if extract_features:
    	    if 'vgg' in model:
    	        predicted_tags, pool1_, pool2_, pool3_, pool4_, pool5_ = tf_out
    	        features['pool1'] = np.concatenate((features['pool1'], np.squeeze(pool1_)), axis=0)
    	        features['pool2'] = np.concatenate((features['pool2'], np.squeeze(pool2_)), axis=0)
    	        features['pool3'] = np.concatenate((features['pool3'], np.squeeze(pool3_)), axis=0)
    	        features['pool4'] = np.concatenate((features['pool4'], np.squeeze(pool4_)), axis=0)
    	        features['pool5'] = np.concatenate((features['pool5'], np.squeeze(pool5_)), axis=0)
    	    else:
    	        predicted_tags, timbral_, temporal_, midend1_, midend2_, midend3_, mean_pool_, max_pool_, penultimate_ = tf_out
    	        features['timbral'] = np.concatenate((features['timbral'], np.squeeze(timbral_)), axis=0)
    	        features['temporal'] = np.concatenate((features['temporal'], np.squeeze(temporal_)), axis=0)
    	        features['cnn1'] = np.concatenate((features['cnn1'], np.squeeze(cnn1_)), axis=0)
    	        features['cnn2'] = np.concatenate((features['cnn2'], np.squeeze(cnn2_)), axis=0)
    	        features['cnn3'] = np.concatenate((features['cnn3'], np.squeeze(cnn3_)), axis=0)
    	        features['mean_pool'] = np.concatenate((features['mean_pool'], mean_pool_), axis=0)
    	        features['max_pool'] = np.concatenate((features['max_pool'], max_pool_), axis=0)
    	        features['penultimate'] = np.concatenate((features['penultimate'], penultimate_), axis=0)
        else:
            predicted_tags = tf_out[0]

        taggram = np.concatenate((taggram, np.array(predicted_tags)), axis=0)

    sess.close()
    print('done!')

    if extract_features:
        return taggram, labels, features
    else:
        return taggram, labels


