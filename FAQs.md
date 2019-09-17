# Frequently Asked Questions (FAQs)

* **Can I run `musicnn` on a CPU?** Yes, the models are already trained.

* **I miss a functionality. How can I get it?** `musicnn` is fairly simple. Feel free to expand it as you wish! Tell us if you think this new functionality is going to be useful for the rest of us.

* **Why `musicnn` contains vgg models?** Because they are a nice baseline, and because people like to use computer vision models for spectrograms. Hence, in this repository you can find `musicnn`-based models (musically motivated convolutional neural networks) and vggs (a computer vision architecture applied to audio).

* **Which is the architecture that `musicnn`-based models employ?** They use a [musically motivated CNN](http://mtg.upf.edu/node/3508) frontend, some [dense layers](https://arxiv.org/abs/1608.06993) in the mid-end, and a [temporal-pooling](https://arxiv.org/abs/1711.02520) back-end. In this [jupyter notebook](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) we provide further details about the model.

* **Which is the best `musicnn` layer-output to pick for transfer learning?** Although we haven't run exhaustive tests, throughout our [visualisations](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) and [preliminary experiments](https://github.com/jordipons/sklearn-audio-transfer-learning) we found the `taggram` and the `max_pool` layer to be the best for this purpose. The `taggram` because it already provides high-level music information, and the `max_pool` layer because it provides a relatively sparse acoustic representation of the music audio.

* **Which 50-tags does the MTT model predict?** These are determined by the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset, that is used for training the MTT models: guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral.

* **Which 50-tags does the MSD model predict?** These are determined by the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split) dataset, that is used for training the MSD models: rock, pop, alternative, indie, electronic, female vocalists, dance, 00s, alternative rock, jazz, beautiful, metal, chillout, male vocalists, classic rock, soul, indie rock, Mellow, electronica, 80s, folk, 90s, chill, instrumental, punk, oldies, blues, hard rock, ambient, acoustic, experimental, female vocalist, guitar, Hip-Hop, 70s, party, country, easy listening, sexy, catchy, funk, electro, heavy metal, Progressive rock, 60s, rnb, indie pop, sad, House, happy.

* **Which are the typical cases where the model fails?** When the input-audio has content that is out of the 50-tags vocabulary. Although in these cases the predictions are consistent and reasonable, the model cannot predict `bass` if this tag is not part of its vocabulary.

* **Why the MTT models predicts `no vocals` and `no vocal`?** Because the vocabulary of the model is determined by the [MagnaTagATune dataset](https://github.com/keunwoochoi/magnatagatune-list) and we used it as it is. 

* **My model is slow, even with a GPU. Can I do something?** Yes! In `./musicnn/configuration.py` you can set a bigger batch size. The dafult is `BATCH_SIZE = 1`, what can be slow – but safe computationally.

* **What are these songs you include in the repository?**  
`./audio/joram-moments_of_clarity-08-solipsism-59-88.mp3` is an electronic music song from the test set of the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset.  
`./audio/TRWJAZW128F42760DD_test.mp3` is an instrumental Muddy Waters song-excerpt from the test set of the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split) called Screamin' And Cryin' (Live In Warsaw 1976).

* **Which audio formats does the `musicnn` library support?** We rely on `librosa` to read audio files. `librosa` uses [soundfile](https://github.com/bastibe/PySoundFile) and [audioread](https://github.com/sampsyo/audioread) for reading audio.
As of v0.7, `librosa` uses soundfile by default, and falls back on audioread only when dealing with codecs unsupported by soundfile (notably, MP3, and some variants of WAV).
For a list of codecs supported by soundfile, see the [libsndfile documentation](http://www.mega-nerd.com/libsndfile/).

* **Which sampling rate, window and hop size were used to compute the log-mel spectrograms?** We compute the STFT of a downsampled signal at 16kHz, with a Hanning window of length 512 (50% overlap). We use 96 mel-bands, and we apply a logarithmic compression to it (`np.log10(10000·x + 1)`).

* **I love this library! How can I send you money?** First, you will need to contact me! [www.jordipons.me/about-me/](http://www.jordipons.me/about-me/)

# Are you using musicnn?
If you are using it for academic works, please cite us:
```
@inproceedings{pons2018atscale,
  title={End-to-end learning for music audio tagging at scale},
  author={Pons, Jordi and Nieto, Oriol and Prockup, Matthew and Schmidt, Erik M. and Ehmann, Andreas F. and Serra, Xavier},
  booktitle={19th International Society for Music Information Retrieval Conference (ISMIR2018)},
  year={2018},
}

```
```
@inproceedings{pons2019musicnn,
  title={musicnn: pre-trained convolutional neural networks for music audio tagging},
  author={Pons, Jordi and Serra, Xavier},
  booktitle={Late-breaking/demo session in 20th International Society for Music Information Retrieval Conference (LBD-ISMIR2019)},
  year={2019},
}

```
If you use it for other purposes, let us know!
