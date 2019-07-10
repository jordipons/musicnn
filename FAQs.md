# Frequently Asked Questions (FAQs)

* **Which 50-tags does the MTT model predict?** These are determined by the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset, that is used for training the MTT model: guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral.

* **Which are the typical cases where the model fails?** When the input audio has content that is out of the 50-tags vocabulary. Although in these cases the predictions are consistent and reasonable, the model cannot predict `bass` if this tag is not part of its vocabulary.

* **Which 50-tags does the MSD model predict?** TO DO.

* **Why the MTT model predicts `no vocals` and `no vocal`?** Because the vocabulary of the model is determined by the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) and we used it as it is. 

* **Why the MSD has 500 and the other 200?** TO DO.

* **My model is slow, even with a GPU. Can I do something?** Yes! In `./musiCNN/configuration.py` you can set a bigger batch size. The dafult is `BATCH_SIZE = 1`, what can be slow – but safe computationally.

* **What are these songs you include in the repository?** `./audio/joram-moments_of_clarity-08-solipsism-59-88.mp3` is an electronic music song from the test set of the MagnaTagATune. `./audio/TRWJAZW128F42760DD_test.mp3` is an instrumental Muddy Waters audio-excerpt from the test set of the Million Song Dataset: Muddy Waters - Screamin' And Cryin' - Live In Warsaw 1976.

* **Which sampling rate, window and hop size where used to compute the log-mel spectrograms?** We compute the STFT of a downsampled signal at 16kHz, with a Hanning window of length 512 (50% overlap). We use 96 mel-bands (computed with librosa defaults), and we apply a logarithmic compression to it (`np.log10(10000 * x + 1)`).

* **I love this library! How can I send you money?** First, contact me on `jordi.pons@dolby.com`.
