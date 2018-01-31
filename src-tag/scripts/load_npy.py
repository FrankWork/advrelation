import numpy as np

npy_file = "embed300.google.npy"

embed_tensor = np.load(npy_file)
print(embed_tensor.shape, embed_tensor.dtype)

# 2.7s