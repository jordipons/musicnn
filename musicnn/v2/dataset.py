import librosa
import numpy as np
import fma.utils as utils
import sklearn as skl
import tensorflow as tf
import pandas as pd
import glob
import time
import random
import re
import tensorflow_io as tfio
from musicnn.v2 import configuration as config


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
        patch = np.expand_dims(audio_rep[time_stamp : time_stamp + n_frames, : ], axis=0).astype(np.float16)
        if first:
            batch = patch
            first = False
        else:
            batch = np.concatenate((batch, patch), axis=0)

    return batch, audio_rep


def create_melspectrogram_dataset(tracks, subdir_name, directory, OUTPUT_DIR, AUDIO_DIR, whole_track=False):


    label_to_id = {name: i for i, name in enumerate(config.GENRES_LABELS)}

    start_time = time.time()
    i = 0
    items = list(tracks.items())

    random.seed(42)
    random.shuffle(items)
    saved_batches_num = 0
    saved_batches = None
    saved_y = None




    def save_dataset_shard():
        output_filename = f"{OUTPUT_DIR}/{subdir_name}/X_{saved_batches_num:06d}.npy"
        np.save(output_filename, saved_batches)

        output_filename = f"{OUTPUT_DIR}/{subdir_name}/y_{saved_batches_num:06d}.npy"
        np.save(output_filename, saved_y)

    for id, val in items:
        i+=1
        filename = utils.get_audio_path(AUDIO_DIR, id)
        n_frames = librosa.time_to_frames(config.INPUT_LENGTH, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
            
        try:
            batch, output_rep  = batch_data(filename, n_frames, n_frames)
        except Exception as e:
            with open("log/dataset.log", "a") as f:
                f.write(f"\nUnused file: {filename}\n")
                f.write(str(e))
            with open("log/unused_files.log", "a") as f:
                f.write(f"Unused file in slice {subdir_name}: {filename}\n")
            continue

        if not whole_track:
            batch = np.expand_dims(batch[4], axis=0)

        new_y = np.full(batch.shape[0], label_to_id[val])

        if saved_batches is None:
            saved_batches = batch
            saved_y = new_y
        else:
            saved_batches = np.concatenate((saved_batches, batch))
            saved_y = np.concatenate((saved_y, new_y))
        

        # np.save(output_filename, batch)
        # directory.loc[len(directory)] = [output_filename, val, subdir_name]

        if i >= 1000:
            save_dataset_shard()
            i = 0
            saved_batches=None
            saved_batches_num+=1
            print(f"{saved_batches_num}: {time.time()-start_time}s")
            start_time = time.time()

    if i != 0:
        save_dataset_shard()



def create_fma_dataset():

    open("dataset.log", "w").close()
    open("unused.log", "w").close()


    tracks = utils.load(f"{config.METADATA_DIR}/tracks.csv")
    # genres = utils.load(f"{METADATA_DIR}/genres.csv")
    # features = utils.load(f"{METADATA_DIR}/features.csv")
    # echonest = utils.load(f"{METADATA_DIR}/echonest.csv")

    small = tracks['set', 'subset'] <= 'small'
    train = tracks['set', 'split'] == 'training'
    val = tracks['set', 'split'] == 'validation'
    test = tracks['set', 'split'] == 'test'

    
    directory = pd.DataFrame({"track": [], "label": [], "split": []})
    y_train = tracks.loc[small & train, ('track', 'genre_top')]
    y_val = tracks.loc[small & val, ('track', 'genre_top')]
    y_test = tracks.loc[small & test, ('track', 'genre_top')]


    create_melspectrogram_dataset(y_train, "train", directory, config.OUTPUT_DIR, config.AUDIO_DIR)
    create_melspectrogram_dataset(y_val, "val", directory, config.OUTPUT_DIR, config.AUDIO_DIR)
    create_melspectrogram_dataset(y_test, "test", directory, config.OUTPUT_DIR, config.AUDIO_DIR)

    directory.to_csv(f"{config.OUTPUT_DIR}/labels.csv", index=False)
    # y_test = tracks.loc[small & test, ('track', 'genre_top')]
    # X_train = features.loc[small & train, 'mfcc']
    # X_test = features.loc[small & test, 'mfcc']

    # X_train, y_train = skl.utils.shuffle(X_train, y_train, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance.
    # scaler = skl.preprocessing.StandardScaler(copy=False)
    # scaler.fit_transform(X_train)
    # scaler.transform(X_test)


def add_label_noise(noisy_labels_ratio, labels):
    # print("--------------")
    # print(np.unique(labels, return_counts=True))
    if noisy_labels_ratio>0:
        idx = list(range(len(labels)))
        random.seed(42)
        random.shuffle(idx)
        num_noise = int(noisy_labels_ratio*len(labels))
        noise_idx = idx[:num_noise]
        for i in range(len(labels)):
            if i in noise_idx:
                noiselabel = random.randint(0, len(config.GENRES_LABELS))
                labels[i] = noiselabel
    # print(np.unique(labels, return_counts=True))
    # print("--------------")



def add_noise_to_fma_dataset(noisy_labels_ratio=0):
    y_files = glob.glob(f"{config.OUTPUT_DIR}/*/y_*.npy")
    for y_file in y_files:
        labels = np.load(y_file)
        add_label_noise(noisy_labels_ratio, labels)
        new_filename = re.sub(r"y_(\d+)\.npy$", rf"ynoisy{int(noisy_labels_ratio * 100)}_\1.npy", y_file)
        np.save(new_filename, labels)



def get_dataset_slice_files(slice_name, noise_ratio=None):
    noise_filename_part=""
    if noise_ratio is not None:
        if noise_ratio <=1:
            noise_percent = int(100 * noise_ratio)
        else:
            noise_percent = noise_ratio
        noise_filename_part = f"noisy{noise_percent}"

    x_files = sorted(glob.glob(f"{config.OUTPUT_DIR}/{slice_name}/X_*.npy"))
    y_files = sorted(glob.glob(f"{config.OUTPUT_DIR}/{slice_name}/y{noise_filename_part}_*.npy"))
    if len(y_files) == 0:
        add_noise_to_fma_dataset(noise_ratio)
        y_files = sorted(glob.glob(f"{config.OUTPUT_DIR}/{slice_name}/y{noise_filename_part}_*.npy"))
    
    return x_files, y_files


def get_dataset_slice(slice_name, noise_ratio=None):


    x_files, y_files = get_dataset_slice_files(slice_name, noise_ratio)
    print(slice_name)
    print(y_files)
    def load_shard(x_path, y_path):
        x = np.load(x_path.decode("utf-8")).astype(np.float16)
        y = np.load(y_path.decode("utf-8"))

        return x, y

    def tf_load(x_path, y_path):
        x, y = tf.numpy_function(
            load_shard,
            [x_path, y_path],
            [tf.float16, tf.int32]
        )
        x.set_shape([None, 187, config.N_MELS])
        y.set_shape([None])
        return x, y

    slice_ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))

    slice_ds = slice_ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    
    slice_ds = slice_ds.unbatch()
    return slice_ds





data_augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomTranslation(0, 0.5, fill_mode="wrap", seed=42),
        ])

def augument_training_data(x):
    TIME_MAX_MASKED_BAND = 50
    FREQ_MAX_MASKED_BAND = 20
    print(x.shape)
    result = data_augmentation_layers(x)

    def my_time_mask(x):
        return tfio.audio.time_mask(x, TIME_MAX_MASKED_BAND)
    result = tf.map_fn(my_time_mask, result)
    
    def my_freq_mask(x):
        return tfio.audio.freq_mask(x, FREQ_MAX_MASKED_BAND)
    result = tf.map_fn(my_freq_mask, result)
    
    return result

def get_dataset(all_noise_ratio = None, train_noise_ratio = None, augument_training = False, batch_size = 1):
    # df = pd.read_csv(f"{config.OUTPUT_DIR}/labels.csv")
    if all_noise_ratio is not None and train_noise_ratio is None:
        train_noise_ratio = all_noise_ratio

    train_ds = get_dataset_slice("train", train_noise_ratio)
    val_ds = get_dataset_slice("val", all_noise_ratio)
    test_ds = get_dataset_slice("test", all_noise_ratio)

    
    train_ds = train_ds.batch(batch_size)

    if augument_training:
        train_ds = train_ds.map(lambda x, y: (augument_training_data(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
        
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = (
        val_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        test_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds

    


import matplotlib.pyplot as plt
def visualize_dataset():
    train, val, test = get_dataset(augument_training=True)
    import tensorflow as tf

    for mel, label in train.take(1):
        break

    print(mel.shape)
    print(config.GENRES_LABELS[int(label)])

    if len(mel.shape) == 3:
        mel = tf.squeeze(mel, axis=0)

    plt.figure(figsize=(10, 4))

    time_axis = np.arange(mel.shape[0]) * config.FFT_HOP / config.SR

    plt.imshow(
        np.transpose(mel),
        origin="lower",
        aspect="auto",
        extent=[0, time_axis[-1], 0, mel.shape[1]]
    )

    plt.colorbar(label="log-mel intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel bins")
    plt.title("Mel Spectrogram (3 seconds)")

    plt.tight_layout()
    plt.show()