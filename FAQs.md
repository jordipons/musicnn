# Frequently Asked Questions (FAQs)
* **Which tags does the MTT model predict?** 'guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral'. These are determined by the [https://github.com/keunwoochoi/magnatagatune-list](MagnaTagATune dataset), that is used for training the MTT model.

* **Which tags does the MSD model predict?** 'rock','pop','alternative','indie','electronic','female vocalists','dance','00s','alternative rock','jazz','beautiful','metal','chillout','male vocalists','classic rock','soul','indie rock','Mellow','electronica','80s','folk','90s','chill','instrumental','punk','oldies','blues','hard rock','ambient','acoustic','experimental','female vocalist','guitar','Hip-Hop','70s','party','country','easy listening','sexy','catchy','funk','electro','heavy metal','Progressive rock','60s','rnb','indie pop','sad','House','happy'. These are determined by the [https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split](Milliion Song Dataset dataset), that is used for training the MSD model.

* **Why the MTT model predicts `no vocals` and `no vocal`?** Because the output-taxonomy of the model is determined by the [https://github.com/keunwoochoi/magnatagatune-list](MagnaTagATune dataset), that is used for training the MTT model, and we used it as it is. 

* **Why the MSD has 500 and the other 200?** TO DO.

* **My model is slow, even with a GPU. Can I do something?** Yes! In `./musiCNN/configuration.py` you can set a bigger batch size. The dafult is `BATCH_SIZE = 1`, what can be slow – but safe computationally.

* **What are these songs you include in the repository?** `./audio/joram-moments_of_clarity-08-solipsism-59-88.mp3` is an electronic music song from the test set of the MagnaTagATune. `./audio/TRWJAZW128F42760DD_test.mp3` is an instrumental Muddy Waters audio-excerpt from the test set of the Million Song Dataset: Muddy Waters - Screamin' And Cryin' - Live In Warsaw 1976.

* **Which sampling rate, window and hop size where used to compute the log-mel spectrograms?** We compute the STFT of a downsampled signal at 16kHz, with a Hanning window of length 512 (50% overlap). We use 96 mel-bands (computed with librosa defaults), and we apply a logarithmic compression to it (`np.log10(10000 * x + 1)`).

* **I love this library! How can I send you money?** First, contact me on `jordi.pons@dolby.com`.
