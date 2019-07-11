# Documentation

```python
tags = musiCNN.tagger.top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=None, verbose=0)
```
Predict the `topN` tags from the music audio in `file_name` file with the selected `model`.  

**Input**
- **file_name:** path to the music file to tag.  
*Data format:* string.  
*Example:* './audio/TRWJAZW128F42760DD_test.mp3'
- **model:** select the music audio tagging model.  
*Data format:* string.  
*Options:* 'MTT' (model trained with the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) dataset).
- **topN:** extract N most likelihood tags according to the selected model.  
*Data format:* integer.  
*Example:* 3
- **input_length:** length (in seconds) of the input spectrogram patches.  
*Data format:* floating point number.  
*Example:* 2.5 TEST THAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
- **input_length:** ammount of overlap (in seconds) of the input spectrogram patches.  
*Data format:* floating point number.  
*Example:* 2.5 TEST THAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
- **verbose:** ammount of printed information.  FIND GOOD WORDING FOR THAT!!!!!!
*Data format:* integer (0 or 1).  
*Options:* 0 (minimal), 1 (tags are printed on the screen). DO THAT!!!!!!!

**Output**
- **tags:** `topN` tags from the music audio in `file_name` file with the selected `model`.    
*Data format:* list.  
*Example:* ['synth', 'techno'] TEST THAT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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


