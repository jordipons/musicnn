import argparse
import numpy as np

from musicnn.extractor import extractor


def top_tags(file_name, model='MTT_musicnn', topN=3, input_length=3, input_overlap=False, print_tags=True, save_tags=False):
    ''' Predict the topN tags of the music-clip in file_name with the selected model.

    INPUT

    - file_name: path to the music file to tag.
    Data format: string.
    Example: './audio/TRWJAZW128F42760DD_test.mp3'
    
    - model: select a music audio tagging model.
    Data format: string.
    Options: 'MTT_musicnn', 'MTT_vgg', 'MSD_musicnn', 'MSD_musicnn_big' or 'MSD_vgg'.
    MTT models are trained with the MagnaTagATune dataset.
    MSD models are trained with the Million Song Dataset.
    To know more about these models, check our musicnn / vgg examples, and the FAQs.
    Important! 'MSD_musicnn_big' is only available if you install from source: python setup.py install.

    - topN: extract N most likely tags according to the selected model.
    Data format: integer.
    Example: 3
    
    - input_length: length (in seconds) of the input spectrogram patches. Set it small for real-time applications.
    Note: This is the length of the data that is going to be fed to the model. In other words, this parameter defines the temporal resolution of the taggram.
    Recommended value: 3, because the models were trained with 3 second inputs.
    Observation: the vgg models do not allow for different input lengths. For this reason, the vgg models' input_length needs to be set to 3. However, musicnn models allow for different input lengths: see this jupyter notebook.
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
    
    - save_tags: Path where to store/save the tags.
    Data format: string.
    Example: 'file_name.tags'

    OUTPUT
    
    tags: topN most likely tags of the music-clip in file_name considering the selected model.
    Data format: list.
    Example: ['synth', 'techno']
    '''

    if 'vgg' in model and input_length != 3:
        raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')
    
    taggram, tags = extractor(file_name, model=model, input_length=input_length, input_overlap=input_overlap, extract_features=False)
    tags_likelihood_mean = np.mean(taggram, axis=0)

    if print_tags:
        print('[' + file_name + '] Top' + str(topN) + ' tags: ')

    if save_tags:
        to = open(save_tags, 'a')   
        to.write(file_name + ',' + model + ',input_length=' + str(input_length) + ',input_overlap=' + str(input_overlap)) 

    topN_tags = []
    for tag_index in tags_likelihood_mean.argsort()[-topN:][::-1]:
        topN_tags.append(tags[tag_index])

        if print_tags:
            print(' - ' + tags[tag_index])

        if save_tags:
            to.write(',' + tags[tag_index])

    if save_tags:
        to.write('\n')
        to.close()
            
    return topN_tags


def parse_args():

    parser = argparse.ArgumentParser(description='Predict the topN tags of the music-clip in file_name with the selected model')

    parser.add_argument('file_name',
                        type=str,
                        help='audio file to process')

    parser.add_argument('-mod', '--model', metavar='',
                        type=str,
                        default='MTT_musicnn',
                        help='select the music audio tagging model to employ (python -m musicnn.tagger music.mp3 --model MTT_musicnn)',
                        required=False)

    parser.add_argument('-n', '--topN', metavar='',
                        type=int,
                        default=3,
                        help='extract N most likely tags according to the selected model (python -m musicnn.tagger music.mp3 --topN 10)',
                        required=False)

    parser.add_argument('-len', '--length', metavar='',
                        type=float,
                        default=3.0,
                        help='length (in seconds) of the input spectrogram patches (python -m musicnn.tagger music.mp3 -len 3.1)',
                        required=False)

    parser.add_argument('-ov', '--overlap', metavar='',
                        type=float,
                        default=False,
                        help='ammount of overlap (in seconds) of the input spectrogram patches (python -m musicnn.tagger music.mp3 -ov 1.0)',
                        required=False)

    parser.add_argument('-p', '--print',
                        default=False, 
                        action='store_true',
                        help='employ --print flag for printing the tags (python -m musicnn.tagger music.mp3 --print)',
                        required=False)

    parser.add_argument('-s', '--save', metavar='',
                        type=str,
                        default=False,
                        help='path where to store/save the tags (python -m musicnn.tagger music.mp3 --save out.tags)',
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
                         input_length=params.length, 
                         input_overlap=params.overlap, 
                         print_tags=params.print,
                         save_tags=params.save)

