# musiCNN
Pronounced as "musician", musiCNN is a pre-trained musically motivated CNN.

## Installation
``` git clone https://github.com/jordipons/musiCNN.git```

``` pip install numpy librosa tqdm ```

```pip install tensorflow``` or ```pip install tensorflow-gpu``` (if you have a GPU)

## Run

From within python, you can run this:
~~~~python
file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
from musiCNN.tagger import top_tags
tags = top_tags(file_name, model='MTT', topN=3)
~~~~
```
techno
electronic
synth
```

or this:

```
file_name = './audio/TRWJAZW128F42760DD_test.mp3.mp3'
from musiCNN.tagger import top_tags
tags = top_tags(file_name, model='MTT', topN=3)
```
 >- guitar
 >- piano
 >- fast
