# Documentation

### Python interface

```python
tags = musicnn.tagger.top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=None, print_tags=True)
```
Predict the `topN` tags of the music-clip in `file_name` with the selected `model`.  

**Input**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* './audio/TRWJAZW128F42760DD_test.mp3'
- **model:** select the music audio tagging model.  
*Data format:* string.  
*Options:* 'MTT' (model trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset).  To know more about our this model, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).
- **topN:** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Example:* 3
- **input_length:** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic%20example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) examples to know more about that.   
*Data format:* floating point number.  
*Example:* 3.1
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Example:* 1
- **print_tags:** set it `True` for printing the tags.  
*Note:* although you don't print the tags, these will be returned by the `musicnn.tagger.top_tags()` function.  
*Data format:* `True` or `False` (boolean).  
*Options:* `False` (for NOT printing the tags), `True` (for printing the tags).  
- **store_tags:** Path where to store the tags.  
*Data format:* string.  
*Options:* 'file_name.tags'  
  
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
- **model:** select the music audio tagging model.  
*Data format:* string.  
*Options:* 'MTT' (model trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset).  To know more about our this model, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).
- **topN:** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Example:* 3
- **input_length:** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic%20example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) examples to know more about that.   
*Data format:* floating point number.  
*Example:* 3.1
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Example:* 1
- **extract_features:** set it `True` for extracting the intermediate representations of the model.  
*Data format:* `True` or `False` (boolean).  
*Options:* `False` (for NOT extracting the features), `True` (for extracting the features).  
  
**Output**
- **taggram:**  expresses the temporal evolution of the tags likelihood.  
*Data format:* 2D np.ndarray (time, tags).  
*Example:* see our [basic](https://github.com/jordipons/musicnn/blob/master/basic%20example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) examples.  
- **tags:** list of tags corresponding to the tag-indices of the taggram.  
*Data format:* list.  
*Example:* see our [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md) page for the complete tags list.
- **features:** a dictionary containing the outputs of the different layers that the selected model has.  
*Data format:* dictionary.  
*Keys*: ['timbral',  'temporal', 'cnn1', 'cnn2', 'cnn3', 'mean_pool', 'max_pool', 'penultimate']  
*Example:* see our [advanced](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) examples.
***************

### Command-line interface

```
python -m musicnn.tagger file_name --model 'MTT' --topN 3 --input_length 3 --input_overlap 3, --print_tags --output file.tags
```
Predict the `topN` tags of the music-clip in `file_name` with the selected `model`.  

**Options**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* ./audio/TRWJAZW128F42760DD_test.mp3
- **model:** select the music audio tagging model.  
*Data format:* string.  
*Options:* 'MTT' (model trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset).  To know more about our this model, check our [advanced example](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) and [FAQs](https://github.com/jordipons/musicnn/blob/master/FAQs.md).  
*Default:* 'MTT'.  
- **topN:** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Default:* 3.  
*Example:* 10  
- **input_length:** length (in seconds) of the input spectrogram patches. Set it small for real-time applications.   
This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our [basic](https://github.com/jordipons/musicnn/blob/master/basic%20example.ipynb) / [advanced](https://github.com/jordipons/musicnn/blob/master/advanced%20example.ipynb) examples to know more about that.   
*Data format:* floating point number.  
*Example:* 3.1
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Note:* Set it considering the `input_length`.  
*Data format:* floating point number.  
*Example:* 1
- **print_tags:** set it `True` for printing the tags.  
*Note:* although you don't print the tags, these will be returned by the `musicnn.tagger.top_tags()` function.  
*Data format:* `True` or `False` (boolean).  
*Options:* `False` (for NOT printing the tags), `True` (for printing the tags).  
- **store_tags:** Path where to store the tags.  
*Data format:* string.  
*Options:* 'file_name.tags'  
  
**Output**
- **tags:** `topN` most likely tags of the music-clip in `file_name` considering the selected `model`.    
*Data format:* list.  
*Example:* ['synth', 'techno']
