import numpy as np
from musiCNN.extractor import extractor


def top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=False, print_tags=True):
    ''' Predict the topN tags of the music-clip in file_name with the selected model.

    INPUT

    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'
        
    - model: select the music audio tagging model.
    Data format: string.
    Options: 'MTT' (model trained with the MagnaTagATune dataset). To know more about our this model, check our advanced example and FAQs.
    topN: extract N most likely tags according to the selected model.
    Data format: integer.
    Example: 3
        
    - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
    This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our basic / advanced examples to know more about that.
    Data format: floating point number.
    Example: 3.1
        
    - input_overlap: ammount of overlap (in seconds) of the input spectrogram patches.
    Note: Set it considering the input_length.
    Data format: floating point number.
    Example: 1
        
    - print_tags: set it True for printing the tags.
    Note: although you don't print the tags, these will be returned by the musiCNN.tagger.top_tags() function.
    Data format: True or False (boolean).
    Options: False (for NOT printing the tags), True (for printing the tags).

    OUTPUT

    - tags: topN most likely tags of the music-clip in file_name considering the selected model.
    Data format: list.
    Example: ['synth', 'techno']
    '''

    taggram, tags = extractor(file_name, model=model, input_length=input_length, input_overlap=input_overlap)
    tags_likelihood_mean = np.mean(taggram, axis=0)

    if print_tags:
        print('[' + file_name + '] Top' + str(topN) + ' tags: ')

    topN_tags = []
    for tag_index in tags_likelihood_mean.argsort()[-topN:][::-1]:
        topN_tags.append(tags[tag_index])

        if print_tags:
            print(' - ' + tags[tag_index])

            
    return topN_tags
