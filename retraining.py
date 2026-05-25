import os
import numpy as np
import librosa

import tensorflow as tf

from musicnn.v2 import models
from musicnn.v2 import configuration as config


from musicnn.v2.extractor import extractor
from musicnn.v2.tagger import top_tags

file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
# tags = top_tags(file_name, model='MTT_musicnn', print_tags=True)
tags = top_tags(file_name, model='MTT_vgg', print_tags=True)
quit()



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


def retrain_on_fma():
    import utils
    import sklearn as sklearn
    tracks = utils.load('data/fma_metadata/tracks.csv')
    genres = utils.load('data/fma_metadata/genres.csv')
    features = utils.load('data/fma_metadata/features.csv')
    echonest = utils.load('data/fma_metadata/echonest.csv')

    small = tracks[tracks['set', 'subset'] <= 'small']
    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'

    filename = utils.get_audio_path(AUDIO_DIR, 2)
    y_train = tracks.loc[small & train, ('track', 'genre_top')]
    y_test = tracks.loc[small & test, ('track', 'genre_top')]
    X_train = features.loc[small & train, 'mfcc']
    X_test = features.loc[small & test, 'mfcc']

    X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    scaler = skl.preprocessing.StandardScaler(copy=False)
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
def retrain(file_name, model='MTT_musicnn', input_length=3, input_overlap=False, extract_features=True):
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
    elif 'GENRES' in model:
        labels = config.GENRES_LABELS
        if model.startswith('GENRES_'):
            model = model[len('GENRES_'):]
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

    with tf.name_scope("model"):

        x = tf.keras.Input(shape=(n_frames, config.N_MELS), dtype=tf.float32, name="x")
        # is_training = tf.keras.Input(shape=(), dtype=tf.bool, name="is_training")

        if 'vgg' in model:
            y, pool1, pool2, pool3, pool4, pool5 = models.define_model(
                x, model, num_classes
            )
        else:
            y, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate = models.define_model(
                x,  model, num_classes
            )
    
        # y_changed = tf.keras.layers.Dense()(y)
        normalized_y = tf.keras.layers.Activation("sigmoid")(y_changed)

    # Add an output layer to y, or to penultimate (maybe in define_model?)

    # tensorflow: loading model

    modelv2 =  tf.keras.Model(inputs=x, outputs=normalized_y, name="modelv2")



    try:

        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model)
        # print(os.path.exists(ckpt_path))   

        ckpt_path = f"{ckpt_path}/"  
        modelv2.load_weights(ckpt_path, by_name=True, skip_mismatch=True).expect_partial()
        
    except Exception as e:
        if model == 'MSD_musicnn_big':
            raise ValueError('MSD_musicnn_big model is only available if you install from source: python setup.py install')
        elif model == 'MSD_vgg':
            raise ValueError('MSD_vgg model is still training... will be available soon! :)')
        else:
            raise e
    

    print('Computing spectrogram (w/ librosa) and tags (w/ tensorflow)..', end =" ")
    batch, spectrogram = batch_data(file_name, n_frames, overlap)





