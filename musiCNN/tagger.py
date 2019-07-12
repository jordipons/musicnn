import numpy as np
from musiCNN.extractor import extractor


def top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=False, print_tags=True):
    # TODO: document

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
