import numpy as np

from musicnn.v2.dataset import create_fma_dataset, get_dataset, add_noise_to_fma_dataset


# create_fma_dataset()
add_noise_to_fma_dataset(0.1)


path = "fma/data/melspectrograms3/train/X_000000.npy"
x = np.load(path)
print("y parameters")
print("shape:", x.shape)
print("dtype:", x.dtype)
print("ndim:", x.ndim)



