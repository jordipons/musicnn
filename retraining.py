import os
import numpy as np
import librosa
import keras
import tensorflow as tf

from musicnn.v2 import models
from musicnn.v2 import configuration as config


from musicnn.v2.extractor import extractor
from musicnn.v2.tagger import top_tags
from musicnn.v2.dataset import get_dataset

# file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
# # tags = top_tags(file_name, model='MTT_musicnn', print_tags=True)
# tags = top_tags(file_name, model='MTT_musicnn', print_tags=True, input_length=29)
# quit()







def retrain(model_name='GENRES_MTT_musicnn', extract_features=True):
    
    input_length = config.INPUT_LENGTH
    # select model


    if 'MTT' in model_name:
        labels = config.MTT_LABELS
    elif 'MSD' in model_name:
        labels = config.MSD_LABELS

    num_classes = len(labels)
    num_classes_new = num_classes
    
    if 'GENRES' in model_name:
        labels_new = config.GENRES_LABELS
        num_classes_new = len(labels_new)
        if model_name.startswith('GENRES_'):
            model_name = model_name[len('GENRES_'):]

    if 'vgg' in model_name and input_length != 3:
        raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')

    # convert seconds to frames
    n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
    overlap = n_frames


    # tensorflow: define the model
    print("Hello")

    with tf.name_scope("model"):

        x = tf.keras.Input(shape=(n_frames, config.N_MELS), dtype=tf.float16, name="x")
        # is_training = tf.keras.Input(shape=(), dtype=tf.bool, name="is_training")

        if 'vgg' in model_name:
            y, pool1, pool2, pool3, pool4, pool5 = models.define_model(
                x, model_name, num_classes
            )
        else:
            y, timbral, temporal, cnn1, cnn2, cnn3, mean_pool, max_pool, penultimate = models.define_model(
                x,  model_name, num_classes
            )
        # normalized_y = tf.keras.layers.Activation("sigmoid")(y)
    print("Model read")
    # Add an output layer to y, or to penultimate (maybe in define_model?)

    # tensorflow: loading model
    orig_model =  tf.keras.Model(inputs=x, outputs=y, name="orig_model")

    try:

        ckpt_path = os.path.join(os.path.dirname(__file__), "musicnn",  model_name)
        # print(os.path.exists(ckpt_path))   

        ckpt_path = f"{ckpt_path}/"  
        orig_model.load_weights(ckpt_path).expect_partial()
        # modelv2.load_weights(ckpt_path, by_name=True, skip_mismatch=True).expect_partial()
        

    except Exception as e:
        if model_name == 'MSD_musicnn_big':
            raise ValueError('MSD_musicnn_big model is only available if you install from source: python setup.py install')
        elif model_name == 'MSD_vgg':
            raise ValueError('MSD_vgg model is still training... will be available soon! :)')
        else:
            raise e
    

    base_model = tf.keras.Model(
        inputs=orig_model.input,
        outputs=orig_model.layers[-2].output
    )
    x_orig = base_model.output
    outputs = tf.keras.layers.Dense(num_classes_new, activation="softmax")(x_orig)
    base_model.trainable = False
    modelv2 = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs
    )

    train_ds, val_ds, test_ds = get_dataset()
    modelv2.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    modelv2.summary()

    print(modelv2.evaluate(train_ds))
    modelv2.fit(
        train_ds,
        epochs=8
    )
    # print(modelv2.evaluate(test_ds))

    
    


if __name__ == '__main__':
    retrain()
    # path = "fma/data/melspectrograms3/train/116709.npy"
    # x = np.load(path)

    # print("shape:", x.shape)
    # print("dtype:", x.dtype)
    # print("ndim:", x.ndim)
    



