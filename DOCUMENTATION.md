# Documentation

TODO: explain musicnn. explain vgg.

### Python interface

```python
tags = musicnn.tagger.top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=None, print=True, save_tags=False)
```
Predict the `topN` tags of the music-clip in `file_name` with the selected `model`.  

**Input**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* './audio/TRWJAZW128F42760DD_test.mp3'
- **model:** select a music audio tagging model.  
*Data format:* string.  
*Options:* `'MTT'`, `'MTT_vgg'`, `'MSD'`, `'MSD_big'` or `'MSD_vgg'`.  
`MTT` models are trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset.  
`MSD` models are trained with the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split).  
To know more about these models, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).
- **topN:** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Example:* 3
- **input_length:** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic_example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) examples to know more about that.  
*Recommended value:* 3, because the models were trained with 3 second inputs.  
*Data format:* floating point number.  
*Example:* 3.1
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Example:* 1.0
- **print:** set it `True` for printing the tags.  
*Note:* although you don't print the tags, these will be returned by the `musicnn.tagger.top_tags()` function.  
*Data format:* boolean.  
*Options:* `False` (for NOT printing the tags), `True` (for printing the tags).  
- **save_tags:** Path where to store/save the tags.  
*Data format:* string.  
*Example:* 'file_name.tags'  
  
**Output**
- **tags:** `topN` most likely tags of the music-clip in `file_name` considering the selected `model`.    
*Data format:* list.  
*Example:* ['synth', 'techno']
***************

```python
taggram, tags, features = musicnn.extractor.extractor(file_name, model='MTT', input_length=3, input_overlap=None, extract_features=False)
```
Extract the `taggram` (the temporal evolution of tags) and `features` (intermediate representations of the model) of the music-clip in `file_name` with the selected `model`.  

**Input**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* './audio/TRWJAZW128F42760DD_test.mp3'
- **model:** select a music audio tagging model.  
*Data format:* string.  
*Options:* `'MTT'`, `'MTT_vgg'`, `'MSD'`, `'MSD_big'` or `'MSD_vgg'`.  
`MTT` models are trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset.  
`MSD` models are trained with the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split).  
To know more about these models, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).
- **input_length:** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic_example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) examples to know more about that.   
*Recommended value:* 3, because the models were trained with 3 second inputs.  
*Data format:* floating point number.  
*Example:* 3.1
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Example:* 1.0
- **extract_features:** set it `True` for extracting the intermediate representations of the model.  
*Data format:* boolean.  
*Options:* `False` (for NOT extracting the features), `True` (for extracting the features).  
  
**Output**
- **taggram:**  expresses the temporal evolution of the tags likelihood.  
*Data format:* 2D np.ndarray (time, tags).  
*Example:* see our [basic](https://github.com/jordipons/musicnn/blob/master/basic_example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) examples.  
- **tags:** list of tags corresponding to the tag-indices of the taggram.  
*Data format:* list.  
*Example:* see our [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md) page for the complete tags list.
- **features:** if `extract_features = True`, it outputs a dictionary containing the activations of the different layers the selected model has.  
*Data format:* dictionary.  
*Keys (musicnn model)*: ['timbral',  'temporal', 'cnn1', 'cnn2', 'cnn3', 'mean_pool', 'max_pool', 'penultimate']  
*Keys (vgg model)*: ['vgg1',  'vgg2', 'vgg3', 'vgg4', 'vgg5']  
*Example:* see our [musicnn](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) and [vgg](https://github.com/jordipons/musicnn/blob/master/vgg_example.ipynb) examples.
***************

### Command-line interface

```
python -m musicnn.tagger file_name --model 'MTT' --topN 3 --length 3 --overlap 3 --print --save file.tags
```
Predict the `topN` tags of the music-clip in `file_name` with the selected `model`.  

**Arguments**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* `python -m musicnn.tagger music.mp3`  
- **--model (-mod):** select a music audio tagging model.  
*Data format:* string.  
*Options:* `'MTT'`, `'MTT_vgg'`, `'MSD'`, `'MSD_big'` or `'MSD_vgg'`.  
`MTT` models are trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset.  
`MSD` models are trained with the [Million Song Dataset](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split).  
To know more about these models, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).
*Default:* MTT  
*Example:* `python -m musicnn.tagger music.mp3 --model MTT`  
- **--topN (-n):** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Default:* 3  
*Example:* `python -m musicnn.tagger music.mp3 --topN 10`  
- **--length (-len):** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic_example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/musicnn_example.ipynb) examples to know more about that.   
*Recommended value:* 3, because the models were trained with 3 second inputs.  
*Data format:* floating point number.  
*Default:* 3.0  
*Example:* `python -m musicnn.tagger music.mp3 -len 3.1`  
- **--overlap (-ov):** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Default:* 3.0  
*Example:* `python -m musicnn.tagger music.mp3 -ov 1.0`  
- **--print (-p):** employ this flag for printing the tags.  
*Data format:* boolean.  
*Example:* `python -m musicnn.tagger music.mp3 --print`  
- **--save  (-s):** Path where to store/save the tags.  
*Data format:* string.  
*Output data format:* csv.  
*Example:* `python -m musicnn.tagger music.mp3 --save out.tags`  

  
**Output**
- **tags:** `topN` most likely tags of the music-clip in `file_name` considering the selected `model`.    
