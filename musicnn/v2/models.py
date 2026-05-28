import tensorflow as tf
from musicnn.v2 import configuration as config



def define_model(x,  model, num_classes):

    if model == 'MTT_musicnn':
        return build_musicnn(x,  num_classes, num_filt_midend=64, num_units_backend=200)

    elif model == 'MTT_vgg':
        return vgg(x,  num_classes, 128)

    elif model == 'MSD_musicnn':
        return build_musicnn(x,  num_classes, num_filt_midend=64, num_units_backend=200)

    elif model == 'MSD_musicnn_big':
        return build_musicnn(x,  num_classes, num_filt_midend=512, num_units_backend=500)

    elif model == 'MSD_vgg':
        return vgg(x,  num_classes, 128)       

    elif model == 'GENRES_musicnn':    
        return build_musicnn_fma(x,  num_classes, num_filt_midend=64, num_units_backend=200)
    else:
        raise ValueError(f'Model {model} not implemented!')



def build_musicnn_fma(x,  num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200):
    y, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate = build_musicnn(x,  num_classes, num_filt_frontend, num_filt_midend, num_units_backend)
    

def build_musicnn(x,  num_classes, num_filt_frontend=1.6, num_filt_midend=64, num_units_backend=200):

    ### front-end ### musically motivated CNN
    frontend_features_list = frontend(x, config.N_MELS, num_filt=1.6, type='7774timbraltemporal')
    # concatnate features coming from the front-end
    frontend_features = tf.concat(frontend_features_list, 2)

    
    ### mid-end ### dense layers
    midend_features_list = midend(frontend_features,  num_filt_midend)
    # dense connection: concatnate features coming from different layers of the front- and mid-end
    
    print("\nSHAPE")
    for i in midend_features_list:
        print(i.shape)

    midend_features = tf.concat(midend_features_list, 2)

    ### back-end ### temporal pooling
    logits, penultimate, mean_pool, max_pool = backend(midend_features,  num_classes, num_units_backend, type='globalpool_dense')

    # [extract features] temporal and timbral features from the front-end
    timbral = tf.concat([frontend_features_list[0], frontend_features_list[1]], 2)
    temporal = tf.concat([frontend_features_list[2], frontend_features_list[3], frontend_features_list[4]], 2)
    # [extract features] mid-end features
    cnn1, cnn2, cnn3 = midend_features_list[1], midend_features_list[2], midend_features_list[3]
    mean_pool = tf.squeeze(mean_pool, [2])
    max_pool = tf.squeeze(max_pool, [2])

    return logits, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate


def frontend(x,  yInput, num_filt, type):

    expand_input = tf.expand_dims(x, 3)
    normalized_input = tf.keras.layers.BatchNormalization()(expand_input)

    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(normalized_input, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=(7, int(0.4 * yInput))
                           )

        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=(7, int(0.7 * yInput))
                           )

    if 'temporal' in type:

        s1 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=(128,1)
                          )

        s2 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=(64,1)
                          )

        s3 = tempo_block(inputs=normalized_input,
                          filters=int(num_filt*32),
                          kernel_size=(32,1)
                          )



    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporal':
        return [f74, f77, s1, s2, s3]


import tensorflow as tf

def timbral_block(inputs, filters, kernel_size,  padding="valid", activation="relu"):

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x, )

    pool = tf.keras.layers.MaxPool2D(
        pool_size=(1, x.shape[2]),
        strides=(1, x.shape[2])
    )(x)

    return tf.squeeze(pool, axis=2)

def tempo_block(inputs, filters, kernel_size,  padding="same", activation="relu"):

    x = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding=padding,
        activation=activation
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x, )

    width = x.shape[2]
    pool = tf.keras.layers.MaxPool2D(
        pool_size=(1, width),
        strides=(1, width)
    )(x)

    return tf.squeeze(pool, axis=2)

def midend(front_end_output,  num_filt):

    front_end_output = tf.expand_dims(front_end_output, axis=3)

    # conv layer 1
    x = tf.pad(front_end_output, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    print(x.shape)
    conv1 = tf.keras.layers.Conv2D(
        filters=num_filt,
        kernel_size=(7, x.shape[2]),
        padding="valid",
        activation="relu"
    )(x)

    bn1 = tf.keras.layers.BatchNormalization()(conv1)

    bn1_t = tf.transpose(bn1, [0, 1, 3, 2])


    # conv layer 2 (residual)
    x2 = tf.pad(bn1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    conv2 = tf.keras.layers.Conv2D(
        filters=num_filt,
        kernel_size=(7, x2.shape[2]),
        padding="valid",
        activation="relu"
    )(x2)

    bn2 = tf.keras.layers.BatchNormalization()(conv2 )

    conv2_t = tf.transpose(bn2, [0, 1, 3, 2])

    res2 = conv2_t + bn1_t

    # conv layer 3 (residual)
    x3 = tf.pad(res2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

    conv3 = tf.keras.layers.Conv2D(
        filters=num_filt,
        kernel_size=(7, x3.shape[2]),
        padding="valid",
        activation="relu"
    )(x3)

    bn3 = tf.keras.layers.BatchNormalization()(conv3, )

    conv3_t = tf.transpose(bn3, [0, 1, 3, 2])

    res3 = conv3_t + res2

    return [front_end_output, bn1_t, res2, res3]


def backend(feature_map, num_classes, output_units, type=None):

    # temporal pooling
    max_pool = tf.reduce_max(feature_map, axis=1)
    mean_pool, var_pool = tf.nn.moments(feature_map, axes=[1])

    tmp_pool = tf.concat([max_pool, mean_pool], axis=2)

    x = tf.keras.layers.Flatten()(tmp_pool)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(output_units, activation="relu")(x)

    bn_dense = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(bn_dense)

    logits = tf.keras.layers.Dense(num_classes, activation=None)(x)

    return logits, bn_dense, mean_pool, max_pool

import tensorflow as tf

def vgg(x, num_classes, num_filters=32):

    x = tf.expand_dims(x, axis=3)

    x = tf.keras.layers.BatchNormalization()(x)

    # Block 1
    x = tf.keras.layers.Conv2D(
        num_filters, (3, 3),
        padding="same",
        activation="relu",
        name="1CNN"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(pool1)

    # Block 2
    x = tf.keras.layers.Conv2D(
        num_filters, (3, 3),
        padding="same",
        activation="relu",
        name="2CNN"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(pool2)

    # Block 3
    x = tf.keras.layers.Conv2D(
        num_filters, (3, 3),
        padding="same",
        activation="relu",
        name="3CNN"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(pool3)

    # Block 4
    x = tf.keras.layers.Conv2D(
        num_filters, (3, 3),
        padding="same",
        activation="relu",
        name="4CNN"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(pool4)

    # Block 5
    x = tf.keras.layers.Conv2D(
        num_filters, (3, 3),
        padding="same",
        activation="relu",
        name="5CNN"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4))(x)

    x = tf.keras.layers.Flatten()(pool5)
    x = tf.keras.layers.Dropout(0.5)(x)

    output = tf.keras.layers.Dense(num_classes, activation=None)(x)

    return output, pool1, pool2, pool3, pool4, pool5