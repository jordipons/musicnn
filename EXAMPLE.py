import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./musically_motivated_CNN/")
from musically_motivated_CNN import musiCNN

#file_name='./audio/hans_christian-phantoms-02-coyotes_dance-88-117.mp3'
file_name='./audio/TRWJAZW128F42760DD_test.mp3'

taggram, tags = musiCNN.predict(file_name)

# From the taggram one can estimate the most popular tags
topN = 3
tags_likelihood_mean = np.mean(taggram, axis=0)
print(file_name)
for tag_index in tags_likelihood_mean.argsort()[-topN:][::-1]:
    print(' - [mean] ' + tags[tag_index])

# And we can also print the taggram, and the mean of it
fig, (ax1, ax2) = plt.subplots(1, 2)

pos = np.arange(len(tags))
ax1.title.set_text('Predicted tags (mean of taggram)')
ax1.bar(pos, tags_likelihood_mean)
ax1.set_xticks(pos)
ax1.set_xticklabels(tags, rotation=90)

ax2.title.set_text('Predicted tags (median of taggram)')
ax2.bar(pos, tags_likelihood_median)
ax2.set_xticks(pos)
ax2.set_xticklabels(tags, rotation=90)

plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(224)
ax3 = fig.add_subplot(222)
ax4 = fig.add_subplot(223)

pos = np.arange(len(tags))
ax1.title.set_text('Taggram')
ax1.imshow(taggram.T, interpolation=None, aspect="auto")
ax1.set_yticks(pos)
ax1.set_yticklabels(tags, fontsize=8)

ax2.title.set_text('Patches embedding')
ax2.imshow(np.flipud(patches_embedding.T), interpolation=None, aspect="auto")

ax3.title.set_text('Spectrogram')
ax3.imshow(np.flipud(spectrogram.T),interpolation=None, aspect="auto")

ax4.title.set_text('Full embedding')
ax4.imshow(np.flipud(full_embedding.T),interpolation=None, aspect="auto")

plt.show()
