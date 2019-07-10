import numpy as np
from musiCNN.extractor import extractor


def top_tags(file_name, model='MTT', topN=3, input_length=3, input_overlap=None):
    # TODO: document

    taggram, tags = extractor(file_name, model=model, input_length=input_length, input_overlap=input_overlap)
    tags_likelihood_mean = np.mean(taggram, axis=0)
    for tag_index in tags_likelihood_mean.argsort()[-topN:][::-1]:
        print(' - ' + tags[tag_index])
