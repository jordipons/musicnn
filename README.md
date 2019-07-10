# musiCNN
Pronounced as "musician", musiCNN is a pre-trained musically motivated CNN.

See [basic](https://github.com/jordipons/musiCNN/blob/master/basic%20example.ipynb) and [advanced](https://github.com/jordipons/musiCNN/blob/master/advanced%20example.ipynb) examples for additional ideas on how to use `musiCNN`.

Do you have questions? Check the [FAQs](https://github.com/jordipons/musiCNN/blob/master/FAQs.md).

## Installation
``` git clone https://github.com/jordipons/musiCNN.git```

``` pip install numpy librosa tqdm ```

```pip install tensorflow``` or ```pip install tensorflow-gpu``` (if you have a GPU)

## Run

From within python, you can run this:
~~~~python
from musiCNN.tagger import top_tags
tags = top_tags('./audio/joram-moments_of_clarity-08-solipsism-59-88.mp3', model='MTT', topN=10)
~~~~
>– techno  
>– electronic  
>– synth  
>– fast  
>– beat  
>– drums  
>– no vocals  
>– no vocal  
>– dance  
>– ambient  

or this:

~~~~python
tags = top_tags('./audio/TRWJAZW128F42760DD_test.mp3.mp3', model='MTT', topN=3)
~~~~
>– guitar  
>– piano  
>– fast

These two music clips are included in the `./audio/` folder of this repository. 

You can listen to those and evaluate `musiCNN` yourself!
