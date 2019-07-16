import argparse
import sys
import numpy as np

from musicnn.extractor import extractor


def top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=False, print_tags=True, store_tags=False):
    ''' Predict the topN tags of the music-clip in file_name with the selected model.

    INPUT

    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'
        
    - model: select the music audio tagging model.
    Data format: string.
    Options: 'MTT' (model trained with the MagnaTagATune dataset). To know more about our this model, check our advanced example and FAQs.

    - topN: extract N most likely tags according to the selected model.
    Data format: integer.
    Example: 3
        
    - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
    This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram. Check our basic / advanced examples to know more about that.
    Data format: floating point number.
    Example: 3.1
        
    - input_overlap: ammount of overlap (in seconds) of the input spectrogram patches.
    Note: Set it considering the input_length.
    Data format: floating point number.
    Example: 1.0
        
    - print_tags: set it True for printing the tags.
    Note: although you don't print the tags, these will be returned by the musicnn.tagger.top_tags() function.
    Data format: boolean.
    Options: False (for NOT printing the tags), True (for printing the tags).

    - store_tags: Path where to store the tags.
    Data format: string.
    Example: 'file_name.tags'

    OUTPUT

    - tags: topN most likely tags of the music-clip in file_name considering the selected model.
    Data format: list.
    Example: ['synth', 'techno']
    '''

    taggram, tags = extractor(file_name, model=model, input_length=input_length, input_overlap=input_overlap)
    tags_likelihood_mean = np.mean(taggram, axis=0)

    if print_tags:
        print('[' + file_name + '] Top' + str(topN) + ' tags: ')

    if store_tags:
        to = open(store_tags, 'a')   
        to.write(file_name + ',' + model + ',input_length=' + str(input_length) + ',input_overlap=' + str(input_overlap)) 

    topN_tags = []
    for tag_index in tags_likelihood_mean.argsort()[-topN:][::-1]:
        topN_tags.append(tags[tag_index])

        if print_tags:
            print(' - ' + tags[tag_index])

        if store_tags:
            to.write(',' + tags[tag_index])

    if store_tags:
        to.write('\n')
        to.close()
            
    return topN_tags


def parse_args():

    parser = argparse.ArgumentParser(description='Predict the topN tags of the music-clip in file_name with the selected model')

    parser.add_argument('file_name',
                        type=str,
                        help='Audio file to process')

    parser.add_argument('-m', '--model', dest='model',
                        type=str,
                        default='MTT',
                        help='Select the music audio tagging model to employ',
                        required=False)

    parser.add_argument('-N', '--topN', dest='topN',
                        type=int,
                        default=3,
                        help='Extract N most likely tags according to the selected model',
                        required=False)

    parser.add_argument('-len', '--input_length', dest='input_length',
                        type=float,
                        default=3.0,
                        help='Length (in seconds) of the input spectrogram patches',
                        required=False)

    parser.add_argument('-ov', '--input_overlap', dest='input_overlap',
                        type=float,
                        default=False,
                        help='Ammount of overlap (in seconds) of the input spectrogram patches',
                        required=False)

    parser.add_argument('-p', '--print_tags', dest='print_tags',
                        default=False, 
                        action='store_true',
                        help='Employ --print_tags (or -p) for printing the tags',
                        required=False)

    parser.add_argument('-o', '--output', dest='output',
                        type=str,
                        default=False,
                        help='Path where to store the tags',
                        required=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # read parameters from command line
    params = parse_args()

    # predict topN tags
    topN_tags = top_tags(params.file_name, 
                         model=params.model, 
                         topN=params.topN, 
                         input_length=params.input_length, 
                         input_overlap=params.input_overlap, 
                         print_tags=params.print_tags,
                         store_tags=params.output)

