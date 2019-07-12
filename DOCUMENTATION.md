# Documentation

```python
tags = musiCNN.tagger.top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=None, print_tags=True)
```
Predict the `topN` tags of the music-clip in `file_name` with the selected `model`.  

**Input**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* './audio/TRWJAZW128F42760DD_test.mp3'
- **model:** select the music audio tagging model.  
*Data format:* string.  
*Options:* 'MTT' (model trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset).
- **topN:** extract N most likely tags according to the selected model.  
*Data format:* integer.  
*Example:* 3
- **input_length:** length (in seconds) of the input spectrogram patches.  
*Data format:* floating point number.  
*Example:* 2.5
- **input_overlap:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Data format:* floating point number.  
*Example:* 2.5
- **print_tags:** set it for printing the tags.
*Note*: although you don't print the tags, these will be returned by the `musiCNN.tagger.top_tags()` function.
*Data format:* boolean (`True` or `False`).  
*Options:* `False` (for NOT printing the tags), `True` (for printing the tags).

**Output**
- **tags:** `topN` tags from the music audio in `file_name` file with the selected `model`.    
*Data format:* list.  
*Example:* ['synth', 'techno']
***************

```python
taggram, tags, features = musiCNN.extractor.extractor(file_name, model='MTT', input_length=3, input_overlap=None, extract_features=False)
```
> Brief description of what it does.
>
>**Input**
>
>**Output**
>- *taggram:* a matrix of..
>- *tags:* a list of..
>- *features:* a dictionary of..


