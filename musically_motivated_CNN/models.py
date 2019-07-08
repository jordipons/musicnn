import numpy as np

from absl import logging
logging._warn_preinit_stderr = 0
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import configuration as config


def define_models(x, is_training, model, num_classes):

    if model == 'MTT':
        out_frontend = frontend(x,
                                is_training,
                                config.N_MELS,
                                num_filt=1.6,
                                type='7774timbraltemporal')
        return backend(out_frontend,
                       is_training,
                       num_classes,
                       num_filt=64,
                       output_units=200,
                       type='globalpool_dense')

    elif model == 'MSD':
        out_frontend = frontend(x,
                                is_training,
                                config.N_MELS,
                                num_filt=1.6,
                                type='7774timbraltemporal')
        return backend(out_frontend,
                       is_training,
                       num_classes,
                       num_filt=64,
                       output_units=200,
                       type='globalpool_dense')


  #################
  ### FRONT END ###
  #################


def frontend(x, is_training, yInput, num_filt, type):

    expanded_layer = tf.expand_dims(x, 3)
    input_layer = tf.layers.batch_normalization(expanded_layer, training=is_training)

    if 'timbral' in type:

        # padding only time domain for an efficient 'same' implementation
        # (since we pool throughout all frequency afterwards)
        input_pad_7 = tf.pad(input_layer, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")

        if '74' in type:
            f74 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.4 * yInput)],
                           is_training=is_training)
        if '77' in type:
            f77 = timbral_block(inputs=input_pad_7,
                           filters=int(num_filt*128),
                           kernel_size=[7, int(0.7 * yInput)],
                           is_training=is_training)


    if 'temporal' in type:

        s1 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[128,1],
                          is_training=is_training)

        s2 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[64,1],
                          is_training=is_training)

        s3 = tempo_block(inputs=input_layer,
                          filters=int(num_filt*32),
                          kernel_size=[32,1],
                          is_training=is_training)


    # choose the feature maps we want to use for the experiment
    if type == '7774timbraltemporal':
        concat_list = [f74, f77, s1, s2, s3]

    elif type == '74timbral':
        concat_list = [f74]

    return tf.concat(concat_list, 2)


def timbral_block(inputs, filters, kernel_size, is_training, padding="valid", activation=tf.nn.relu,
                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

    conv = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=kernel_initializer)
    bn_conv = tf.layers.batch_normalization(conv, training=is_training)
    pool = tf.layers.max_pooling2d(inputs=bn_conv,
                                   pool_size=[1, bn_conv.shape[2]],
                                   strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])


def tempo_block(inputs, filters, kernel_size, is_training, padding="same", activation=tf.nn.relu,
                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

    conv = tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            padding=padding,
                            activation=activation,
                            kernel_initializer=kernel_initializer)
    bn_conv = tf.layers.batch_normalization(conv, training=is_training)
    pool = tf.layers.max_pooling2d(inputs=bn_conv,
                                   pool_size=[1, bn_conv.shape[2]],
                                   strides=[1, bn_conv.shape[2]])
    return tf.squeeze(pool, [2])


  ################
  ### BACK END ###
  ################


def backend(route_out, is_training, num_classes, num_filt, output_units, type):
    features = midend(route_out, is_training, num_classes, num_filt, type)
    summarized_features, logits = temporal_pool(features, is_training, num_classes, output_units, type)
    return logits, summarized_features, features


def midend(route_out, is_training, num_classes, num_filt, type):
    route_out = tf.expand_dims(route_out, 3)

    # conv layer 1 - adapting dimensions
    route_out_pad = tf.pad(route_out, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv1 = tf.layers.conv2d(inputs=route_out_pad,
                             filters=num_filt,
                             kernel_size=[7, route_out.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    bn_conv1_t = tf.transpose(bn_conv1, [0, 1, 3, 2])

    # conv layer 2 - residual connection
    bn_conv1_pad = tf.pad(bn_conv1_t, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv2 = tf.layers.conv2d(inputs=bn_conv1_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv1_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv2 = tf.layers.batch_normalization(conv2, training=is_training)
    conv2 = tf.transpose(bn_conv2, [0, 1, 3, 2])
    res_conv2 = tf.add(conv2, bn_conv1_t)

    # conv layer 3 - residual connection
    bn_conv2_pad = tf.pad(res_conv2, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT")
    conv3 = tf.layers.conv2d(inputs=bn_conv2_pad,
                             filters=num_filt,
                             kernel_size=[7, bn_conv2_pad.shape[2]],
                             padding="valid",
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_conv3 = tf.layers.batch_normalization(conv3, training=is_training)
    conv3 = tf.transpose(bn_conv3, [0, 1, 3, 2])
    res_conv3 = tf.add(conv3, res_conv2)

    # which layers?
    if 'dense' in type:
        return tf.concat([route_out, bn_conv1_t, res_conv2, res_conv3], 2)
    else:
        return res_conv3


def temporal_pool(feature_map, is_training, num_classes, output_units, type):

    max_pool = tf.reduce_max(feature_map, axis=1)
    avg_pool, var_pool = tf.nn.moments(feature_map, axes=[1])
    tmp_pool = tf.concat([max_pool, avg_pool], 2)

    # output - 2 dense layer with droupout
    flat_pool = tf.contrib.layers.flatten(tmp_pool)
    flat_pool = tf.layers.batch_normalization(flat_pool, training=is_training)
    flat_pool_dropout = tf.layers.dropout(flat_pool, rate=0.5, training=is_training)
    dense = tf.layers.dense(inputs=flat_pool_dropout,
                            units=output_units,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    bn_dense = tf.layers.batch_normalization(dense, training=is_training)
    dense_dropout = tf.layers.dropout(bn_dense, rate=0.5, training=is_training)
    logits = tf.layers.dense(inputs=dense_dropout,
                           activation=None,
                           units=num_classes,
                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer())

    return bn_dense, logits


